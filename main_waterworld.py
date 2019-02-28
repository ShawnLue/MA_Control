import argparse
import torch as th
import datetime
import pathlib
import ruamel.yaml as yaml
from types import SimpleNamespace as SN
import os
import shutil
import time
import visdom
import numpy as np
from gym.spaces import Box
from pathlib import Path
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with open('./config_waterworld.yaml', 'r') as f:
    args_dict = yaml.safe_load(f)
    args_dict['n_agents'] = args_dict['n_pursuers']
    # config = yaml.load(f, Loader=loader)
    args = SN(**args_dict)

unique_token = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
result_folder = './results/' + args.task + '/' + args.alg + '/' + unique_token
pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
shutil.copy('./config_waterworld.yaml', result_folder + '/config_waterworld.yaml')

model_path = result_folder + '/model'
film_path = result_folder + '/film'
pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(film_path).mkdir(parents=True, exist_ok=True)

vis = visdom.Visdom(port=args.port, env='continuous-waterworld_{}_{}'.format(args.alg, unique_token),
                    use_incoming_socket=False)


def make_parallel_env(**kwargs):
    def get_env_fn(rank):
        def init_env():
            env = make_env(**kwargs)
            env.seed(kwargs['seed'] + rank * 1000)
            np.random.seed(kwargs['seed'] + rank * 1000)
            return env
        return init_env
    if kwargs['n_rollout_threads'] == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(kwargs['n_rollout_threads'])])


def run(args, **args_dict):
    reward_flag, pos_flag = None, None
    save_data = {'reward': 500.}
    # model_dir = Path('./models') / config.env_id / config.model_name
    # if not model_dir.exists():
    #     curr_run = 'run1'
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                      model_dir.iterdir() if
    #                      str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         curr_run = 'run1'
    #     else:
    #         curr_run = 'run%i' % (max(exst_run_nums) + 1)
    # run_dir = model_dir / curr_run
    # log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.use_cuda or not th.cuda.is_available():
        # th.set_num_threads(args.n_training_threads)
        FloatTensor = th.FloatTensor
    else:
        FloatTensor = th.cuda.FloatTensor
    env = make_parallel_env(**args_dict)
    maddpg = MADDPG.init_from_env(env, args)
    print(env.observation_space[0].shape, env.action_space[0].shape)
    replay_buffer = ReplayBuffer(args.capacity, args.n_agents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, args.n_episodes, args.n_rollout_threads):
        ttt = time.time()
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        if args.use_cuda and th.cuda.is_available():
            maddpg.prep_rollouts(device='gpu')
        else:
            maddpg.prep_rollouts(device='cpu')
        # maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, args.n_exploration_eps - ep_i) / args.n_exploration_eps
        scale_noise_i = args.final_noise_scale + (args.init_noise_scale - args.final_noise_scale) * explr_pct_remaining
        maddpg.scale_noise(scale_noise_i)
        maddpg.reset_noise()

        print("Episodes %i-%i of %i, replay: %.2f, explore: %.2f" % (ep_i + 1,
                                                                     ep_i + 1 + args.n_rollout_threads,
                                                                     args.n_episodes,
                                                                     float(len(replay_buffer)) / replay_buffer.max_steps,
                                                                     scale_noise_i))

        for et_i in range(args.max_steps):
            ttt = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [th.from_numpy(np.vstack(obs[:, i])).type(FloatTensor) for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(args.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += args.n_rollout_threads
            #
            # ttt2 = time.time()
            # print('1', ttt2 - ttt)
            #
            if (len(replay_buffer) >= args.batch_size and
                    (t % args.steps_per_update) < args.n_rollout_threads):
                ttt = time.time()
                if args.use_cuda and th.cuda.is_available():
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                # for u_i in range(args.n_rollout_threads):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(args.batch_size,
                                                  to_gpu=args.use_cuda and th.cuda.is_available(),
                                                  norm_rews=args.norm_rews)
                    _, _, _ = maddpg.update(sample, a_i)
                maddpg.update_all_targets()
                if args.use_cuda and th.cuda.is_available():
                    maddpg.prep_rollouts(device='gpu')
                else:
                    maddpg.prep_rollouts(device='cpu')
                # maddpg.prep_rollouts(device='cpu')
                #
                # ttt2 = time.time()
                # print('2', ttt2 - ttt)
                #
        # ep_rews = replay_buffer.get_average_rewards(
        #     config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % args.test_interval < args.n_rollout_threads:
            # ttt = time.time()
            obs = env.reset()
            if args.use_cuda and th.cuda.is_available():
                maddpg.prep_rollouts(device='gpu')
            else:
                maddpg.prep_rollouts(device='cpu')
            # maddpg.prep_rollouts(device='cpu')
            with th.no_grad():
                collision_total = np.zeros((args.n_rollout_threads, 2))
                record_collision = np.zeros(2)
                finish_ep = np.zeros(args.n_rollout_threads)
                r_total = np.zeros((args.n_rollout_threads, args.n_agents))
                record_r = np.zeros(args.n_agents)
                for eval_i in range(args.max_steps):
                    torch_obs = [FloatTensor(np.vstack(obs[:, i])) for i in range(maddpg.nagents)]
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
                    agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]
                    actions = [[ac[i] for ac in agent_actions] for i in range(args.n_rollout_threads)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    r_total += rewards
                    collision_total += infos
                    obs = next_obs
                    for d_i in range(dones.shape[0]):
                        if dones[d_i] or (eval_i == args.max_steps - 1 and finish_ep[d_i] == 0.):
                            # if eval_i == args.max_steps - 1 and finish_ep[d_i] == 0.:
                            #     print(d_i)
                            record_r += r_total[d_i]
                            r_total[d_i] = [0.] * args.n_agents
                            record_collision += collision_total[d_i]
                            collision_total[d_i] = [0] * 2
                            finish_ep[d_i] += 1
                record_r /= finish_ep.sum()
                record_collision /= finish_ep.sum()

                # ttt2 = time.time()
                # print('3', ttt2 - ttt)
                #

                new_path = model_path + '/' + str(ep_i) + '_' + "{:.2f}".format(record_r.sum()) + '.pt'
                if record_r.sum() > save_data['reward']:
                    save_data['reward'] = record_r.sum()
                    if save_data['reward'] > 1000.:
                        # pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
                        maddpg.save(new_path)

                if reward_flag is None:
                    reward_flag = vis.line(X=np.arange(ep_i, ep_i + 1),
                                           Y=np.array([np.append(record_r, record_r.sum())]),
                                           opts=dict(
                                               ylabel='Test Reward',
                                               xlabel='Episode',
                                               title='Reward',
                                               legend=['Agent-%d' % i for i in range(args.n_agents)] + ['Total']
                                           ))
                else:
                    vis.line(X=np.array([np.array(ep_i).repeat(args.n_agents + 1)]),
                             Y=np.array([np.append(record_r, record_r.sum())]),
                             win=reward_flag,
                             update='append')

                if pos_flag is None:
                    pos_flag = vis.line(X=np.arange(ep_i, ep_i + 1),
                                        Y=np.array([record_collision]),
                                        opts=dict(
                                            ylabel='Collision',
                                            xlabel='Episode',
                                            title='Collision',
                                            legend=['ball', 'poison']
                                        ))
                else:
                    vis.line(X=np.array([np.array(ep_i).repeat(2)]),
                             Y=np.array([record_collision]),
                             win=pos_flag,
                             update='append')
        # if ep_i % config.save_interval < config.n_rollout_threads:
        #     os.makedirs(run_dir / 'incremental', exist_ok=True)
        #     maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        #     maddpg.save(run_dir / 'model.pt')

    # maddpg.save(run_dir / 'model.pt')
    env.close()
    # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    # logger.close()


if __name__ == '__main__':

    # parser.add_argument("--n_episodes", default=25000, type=int)
    # parser.add_argument("--episode_length", default=25, type=int)
    # parser.add_argument("--steps_per_update", default=100, type=int)
    # parser.add_argument("--batch_size",
    #                     default=1024, type=int,
    #                     help="Batch size for model training")
    # parser.add_argument("--n_exploration_eps", default=25000, type=int)
    # parser.add_argument("--save_interval", default=1000, type=int)
    # parser.add_argument("--hidden_dim", default=64, type=int)
    # parser.add_argument("--lr", default=0.01, type=float)
    # parser.add_argument("--tau", default=0.01, type=float)
    # parser.add_argument("--agent_alg",
    #                     default="MADDPG", type=str,
    #                     choices=['MADDPG', 'DDPG'])
    #
    # config = parser.parse_args()

    run(args, **args_dict)
