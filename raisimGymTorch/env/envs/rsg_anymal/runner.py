from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_anymal
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import sys
import math
import time
import matplotlib.pyplot as plt
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

#argument
test_mode = sys.argv[1] == 'True'

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
curriculum_start = cfg['environment']['curriculum']['curriculum_start']

# create environment from the configuration file
if test_mode:
    cfg_tmp = cfg
    cfg_tmp['environment']['num_envs'] = 1
    env = VecEnv(rsg_anymal.RaisimGymEnv(task_path + "/anymal", dump(cfg_tmp['environment'] , Dumper=RoundTripDumper)), cfg['environment'])
else:
    env = VecEnv(rsg_anymal.RaisimGymEnv(task_path + "/anymal", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/data",
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/runner.py"])

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []
avg_dones = []
fig, ax = plt.subplots(1, 2, constrained_layout=True, sharex=True, figsize=[10.8, 4.8])

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'],
                                        nn.LeakyReLU,
                                        ob_dim,
                                        act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         'cuda')

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'],
                                          nn.LeakyReLU,
                                          ob_dim,
                                          1),
                           'cuda')

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device='cuda',
              log_dir=saver.data_dir,
              mini_batch_sampling='in_order',
              )

if not test_mode:
    for update in range(1000000):
        ax[0].set(xlabel='iteration', ylabel='avg performance', title='average performance')
        ax[1].set(xlabel='iteration', ylabel='avg dones', title='average dones')
        ax[0].grid()
        ax[1].grid()

        start = time.time()
        env.reset()
        reward_ll_sum = 0
        done_sum = 0
        average_dones = 0.

        if update % cfg['environment']['eval_every_n'] == 0:
            print("Visualizing and evaluating the current policy")
            actor.save_deterministic_graph(saver.data_dir+"/policy_"+str(update)+'.pt', torch.rand(1, ob_dim).cpu())

            parameters = np.zeros([0], dtype=np.float32)
            for param in actor.deterministic_parameters():
                parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
            np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
            loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt')

            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

            for step in range(n_steps*2):
                time.sleep(0.01)
                obs = env.observe(False)
                action_ll = loaded_graph(torch.from_numpy(obs).cpu())
                reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

            env.stop_video_recording()
            env.turn_off_visualization()

            env.reset()
            # model.save(saver.data_dir+"/policies/policy", update)
            env.save_scaling(saver.data_dir, str(update))

        # actual training
        for step in range(n_steps):
            obs = env.observe()
            action = ppo.observe(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[])
            done_sum = done_sum + sum(dones)
            reward_ll_sum = reward_ll_sum + sum(reward)

        # take st step to get value obs
        obs = env.observe()
        ppo.update(actor_obs=obs,
                   value_obs=obs,
                   log_this_iteration=update % 10 == 0,
                   update=update)

        if update > curriculum_start:
            env.curriculum_callback()

        end = time.time()

        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps

        avg_dones.append(average_dones)
        avg_rewards.append(average_ll_performance)

        ax[0].plot(range(len(avg_rewards)), avg_rewards)
        ax[1].plot(range(len(avg_dones)), avg_dones)


        fig.savefig(saver.data_dir + '/demo.png', bbox_inches='tight')

        ax[0].clear()
        ax[1].clear()

        actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to('cuda'))

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('std: ')
        print(np.exp(actor.distribution.std.cpu().detach().numpy()))
        print('----------------------------------------------------\n')

if test_mode:
    curriculum_setting = True
    save_dir = os.environ['WORKSPACE'] + "/ME491TermProject/data/~~~~~~~~"
    test_policy = 1000
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(test_policy)+'.mp4')
    env.load_scaling(save_dir, test_policy)
    if curriculum_setting:
        if test_policy > curriculum_start:
            for i in range(test_policy - curriculum_start):
                env.curriculum_callback()

    loaded_graph = torch.jit.load(save_dir + "/policy_" + str(test_policy) + '.pt')
    print("load_graph")
    dones = False
    steps = 0
    env.reset()
    time.sleep(1)
    while (not dones) and steps < 2 * n_steps:
        obs = env.observe(False)
        action_ll = loaded_graph(torch.from_numpy(obs).cpu())
        env.step(action_ll.cpu().detach().numpy())
        time.sleep(0.01)
        steps += 1
        if steps == 2 * n_steps:
            print("success!")

    env.stop_video_recording()
    env.turn_off_visualization()
    env.close()