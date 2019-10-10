import numpy as np
import torch
import pdb

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_from_gym_env


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        pdb.set_trace()
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def evaluate_gym_env(actor_critic, ob_rms, envs, seed, num_processes, eval_log_dir,
             device, num_steps = 10):

    print("Running evaluation \n")
    envs.envs[0].set_is_binary_reward(True)

    vec_norm = utils.get_vec_normalize(envs)
    
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_steps:
        with torch.no_grad():
            # pdb.set_trace()
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                print("Evaluation| Episode number: {}".format(len(eval_episode_rewards)+1))
                eval_episode_rewards.append(info['episode']['r'])

    # envs.close()
    envs.envs[0].set_is_binary_reward(False)

    f = open(eval_log_dir + "stats.txt", "w+")
    print("Evaluation using {} episodes: mean reward {:.5f} | median reward {:.5f} | min reward {:.5f} | max reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.median(eval_episode_rewards), np.min(eval_episode_rewards), np.max(eval_episode_rewards)))
    f.write("Evaluation using {} episodes: mean reward {:.5f} | median reward {:.5f} | min reward {:.5f} | max reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards), np.median(eval_episode_rewards), np.min(eval_episode_rewards), np.max(eval_episode_rewards)))
    f.close()

    return eval_episode_rewards


