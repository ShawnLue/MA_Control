from madrl_environments.pursuit import MAWaterWorld_mod as Env
from algorithms.maddpg import MADDPG
import numpy as np
import torch as th
import imageio
import itertools

import ruamel.yaml as yaml
from types import SimpleNamespace as SN

unique_token = "20190216-165936"

result_folder = './results/waterworld/MADDPG/' + unique_token
pool_list = ['6384_2965.55']
config_file = result_folder + '/config_waterworld.yaml'

with open(config_file, 'r') as f:
    args_dict = yaml.safe_load(f)
    args_dict['n_agents'] = args_dict['n_pursuers']
    # config = yaml.load(f, Loader=loader)
    args = SN(**args_dict)

if args.seed is not False:
    th.manual_seed(args.seed)

env = Env(**args_dict)

if args.seed is not False:
    env.seed(args.seed)

for m in pool_list:
    maddpg = MADDPG.init_from_save(result_folder + '/model/' + m + '.pt', with_cpu=True)
    maddpg.prep_rollouts(device='cpu')
    with th.no_grad():
        total_reward = 0.
        test_time = 5
        for it in range(test_time):
            l = []
            obs = env.reset()
            l.append(env.render(gui=True))
            obs = np.stack(obs, axis=0)
            obs = th.from_numpy(obs).float()
            print('----------')
            reward_it = 0.
            for t in range(args.test_max_steps):
                obs = obs.type(th.FloatTensor)
                actions = maddpg.step(obs, explore=False)
                agent_actions = [ac.detach().cpu().numpy() for ac in actions]
                next_obs, rewards, dones, _ = env.step(agent_actions)
                next_obs = np.stack(next_obs, axis=0)
                next_obs = th.from_numpy(next_obs)
                reward_it += rewards
                obs = next_obs
                l.append(env.render(gui=True))
                if dones:
                    break
            print(reward_it)
            total_reward += reward_it
            # imageio.mimsave(result_folder + '/film/' + str(m) + '-' + str(it) + '.gif', l, duration=0.01)
        print(m, total_reward / test_time)