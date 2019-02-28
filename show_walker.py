from madrl_environments.walker.multi_walker import MultiWalkerEnv as Env
from algorithms.maddpg import MADDPG
import numpy as np
import torch as th
import imageio
import itertools

import ruamel.yaml as yaml
from types import SimpleNamespace as SN

unique_token = "20190215-143422"
# unique_token_list = ["20190215-143422", "20190215-115109"]
# pool_list_list = [[500, 6220, 28980, 34340], [340, 2700, 11260, 18980]]
# pool_list_list = [[500, 6220, 12820, 28980, 34340], [340, 2700, 11260, 18980]]
# marker_dict = {500: 'bs', 6220: 'w', 12820: 'w', 28980: 'bv', 34340: 'w', 340: 'bo', 2700: 'rv', 11260: 'ro', 18980: 'rs'}
# pool_list_list = [[500, 6220, 34340], [340, 2700, 18980]]
# legend_dict = {500: 'MADDPG-5k', 28980: 'MADDPG-0k', 340: 'MADDPG-10k', 2700: 'FLE-0k', 11260: 'FLE-10k', 18980: 'FLE-5k'}


result_folder = './results/walker/MADDPG/' + unique_token
pool_list = [500, 6220, 12820, 28980, 34340]
# pool_list = [340, 2700, 11260, 18980]
config_file = result_folder + '/config_walker.yaml'

with open(config_file, 'r') as f:
    args_dict = yaml.safe_load(f)
    args_dict['n_agents'] = args_dict['n_walkers']
    # config = yaml.load(f, Loader=loader)
    args = SN(**args_dict)

if args.seed is not False:
    th.manual_seed(args.seed)

env = Env(**args_dict)

if args.seed is not False:
    env.seed(args.seed)

for m in pool_list:
    maddpg = MADDPG.init_from_save(result_folder + '/model/' + str(m) + '.pt', with_cpu=True)
    maddpg.prep_rollouts(device='cpu')
    with th.no_grad():
        total_reward, total_pos = 0., 0.
        test_time = 1
        for it in range(test_time):
            l = []
            obs = env.reset()
            l.append(env.render(mode='rgb_array'))
            obs = np.stack(obs, axis=0)
            obs = th.from_numpy(obs).float()
            print('----------')
            for t in range(args.test_max_steps):
                obs = obs.type(th.FloatTensor)
                actions = maddpg.step(obs, explore=False)
                agent_actions = [ac.detach().cpu().numpy() for ac in actions]
                next_obs, rewards, dones, pos = env.step(agent_actions)
                next_obs = np.stack(next_obs, axis=0)
                next_obs = th.from_numpy(next_obs)
                total_reward += rewards
                # array_pool.append(np.concatenate([obs.numpy().flatten(), np.array(agent_actions).flatten(), [total_reward.sum()]]))
                # if t == 20:
                #     np.save(str(m) + '.npy', array_pool)
                #     total_reward = 0.
                #     break
                obs = next_obs
                l.append(env.render(mode='rgb_array'))
                if dones:
                    print(it, t)
                    total_pos += pos['pos']
                    break
            print('----------')
            # imageio.mimsave(result_folder + '/film/' + str(m) + '-' + str(it) + '.gif', l, duration=0.01)
            if t == args.max_steps - 1:
                total_pos += pos['pos']
        print(m, total_reward / test_time, total_pos / test_time)

# import matplotlib.pyplot as plt
#
# d = []
#
# for i in pool_list_list:
#     for j in i:
#         d.append(np.load(str(j) + '.npy'))
#
# from sklearn.manifold import TSNE
#
# tsne = TSNE(n_components=2, init='pca', random_state=0)
#
# f1 = lambda x: x[:, :-1]
# f2 = lambda x: [x[:, -1][-1]] * x.shape[0]
#
#
# data = np.concatenate(list(map(f1, d)), axis=0)
# value = np.concatenate(list(map(f2, d)))
# label = np.repeat(np.arange(len(pool_list_list[0]) + len(pool_list_list[1])), d[0].shape[0])
#
# result = tsne.fit_transform(data)
#
# x_min, x_max = result.min(axis=0)
# y_min, y_max = result.max(axis=0)
#
# x_norm = (result - x_min) / (x_max - x_min)
# fig, ax = plt.subplots(1)
# fig.set_size_inches(10,8)
#
# # print(label)
# # markers = itertools.cycle(('+', '*', ',', 'o', '.', '1', 'p', '^', 'v', 's', 'X'))
# markers = itertools.cycle(('p', '*', 'o'))
# colors = itertools.cycle(('b', 'g', 'r'))
# # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k'))
#
# # for i in range(len(pool_list_list[0]) + len(pool_list_list[1])):
# # print(x_norm.shape[0])
# distance_list = []
# fontsize=20
# markersize=15
# for i in range(len(pool_list_list[0])):
#     print(i)
#     ind = pool_list_list[0][i]
#     co = marker_dict[ind]
#     if co != 'w':
#         x1 = x_norm[i * 21:(i + 1) * 21, 0]
#         y1 = x_norm[i * 21:(i + 1) * 21, 1]
#         if ind == 340:
#             x1 += np.random.randint(-5, 5)
#             y1 += np.random.randint(-5, 5)
#         ax.plot(x1, y1, co, label=legend_dict[ind], markersize=markersize)
#
# markers2 = itertools.cycle(('^', 'v', 's'))
# colors2 = itertools.cycle(('c', 'm', 'y'))
# for i in range(len(pool_list_list[0]), len(pool_list_list[0]) + len(pool_list_list[1])):
#     print(i)
#     ind = pool_list_list[1][i - len(pool_list_list[0])]
#     co = marker_dict[ind]
#     if co != 'w':
#         x1, y1 = x_norm[i * 21:(i + 1) * 21, 0], x_norm[i * 21:(i + 1) * 21, 1]
#         if ind == 340:
#             randx = np.random.randint(-5, 5, 21)
#             randy = np.random.randint(-5, 5, 21)
#             x1 += randx
#             y1 += (randy - 50)
#         ax.plot(x1, y1, co, label=legend_dict[ind], markersize=markersize)
#
#     # plt.plot(x_norm[i * 21:(i + 1) * 21, 0], x_norm[i * 21:(i + 1) * 21, 1], label[i])
# handles, labels = ax.get_legend_handles_labels()
# handles = [handles[1], handles[0], handles[2], handles[3], handles[5], handles[4]]
# labels = [labels[1], labels[0], labels[2], labels[3], labels[5], labels[4]]
# ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1.0, 0.2501), ncol=2, fancybox=True, shadow=False)
# # ax.legend(handles, labels, fontsize=fontsize, loc=9, ncol=2, fancybox=True, shadow=False)
# # ax.set_aspect('equal')
# '''
# for i in range(x_norm.shape[0]):
#     # print(x_norm[i, 0], x_norm[i, 1])
#     plt.text(x_norm[i, 0], x_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# '''
# plt.xticks([])
# plt.yticks([])
# plt.title('Trajectory embedding in Multi-Walker.', fontsize=fontsize)
# # plt.show()
# plt.savefig('tsne.pdf', bbox_inches='tight')
