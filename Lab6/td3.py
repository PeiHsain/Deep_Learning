'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        # 8 observation -> Q-value of 2 action
        h1, h2 = hidden_dim
        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(h2, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        ## TODO ##
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        return output


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        # 2 action -> Q-value of 2 action
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class TD3:
    def __init__(self, args):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net1 = CriticNet().to(args.device)
        self._critic_net2 = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net1 = CriticNet().to(args.device)
        self._target_critic_net2 = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net1.load_state_dict(self._critic_net1.state_dict())
        self._target_critic_net2.load_state_dict(self._critic_net2.state_dict())
        ## TODO ##
        # choose the optimizer
        self._actor_opt = torch.optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt1 = torch.optim.Adam(self._critic_net1.parameters(), lr=args.lrc)
        self._critic_opt2 = torch.optim.Adam(self._critic_net2.parameters(), lr=args.lrc)

        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.noise_clip = args.noise_clip
        self.action_limit = args.action_limit
        self.policy_delay = args.policy_delay

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        # change state from np.array(8) cpu to tensor(1, 8) gpu 
        state = torch.tensor(state).view(1, -1).to(self.device)
        with torch.no_grad():
            if noise:
                # add the exploration noise
                # make action in [-action_limit, action_limit]
                action = self._actor_net(state).clamp(-self.action_limit, self.action_limit)
                # make noise in [-c, c]
                explo_noise = torch.tensor(self._action_noise.sample().astype('float32')).view(1, -1).to(self.device)
                explo_noise = torch.clamp(explo_noise, -self.noise_clip, self.noise_clip)
                return (action + explo_noise).cpu().numpy().squeeze()
            else:
                return self._actor_net(state).cpu().numpy().squeeze()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self, epoch):
        # update the behavior networks
        self._update_behavior_network(self.gamma, epoch)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net1, self._critic_net1,
                                    self.tau)
        self._update_target_network(self._target_critic_net2, self._critic_net2,
                                    self.tau)

    def _update_behavior_network(self, gamma, epoch):
        actor_net, critic_net1, critic_net2, target_actor_net, target_critic_net1, target_critic_net2 = self._actor_net, self._critic_net1, self._critic_net2, self._target_actor_net, self._target_critic_net1, self._target_critic_net2
        actor_opt, critic_opt1, critic_opt2 = self._actor_opt, self._critic_opt1, self._critic_opt2

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        # calculate q_value and q_target to get loss
        q_value1 = critic_net1(state, action)
        q_value2 = critic_net2(state, action)
        with torch.no_grad():
            ## Target Policy Smoothing -> add noise to target action
            noise = torch.rand_like(action).to(self.device).clamp(-self.noise_clip, self.noise_clip)
            a_next = (target_actor_net(next_state) + noise).clamp(-self.action_limit, self.action_limit)
            ## Clipped Double Q Learning -> pick smaller Q value from two critic net
            q_next1 = target_critic_net1(next_state, a_next)
            q_next2 = target_critic_net2(next_state, a_next)
            q_next = torch.min(q_next1, q_next2)
            # if done=True, don't have next state
            q_target = reward + gamma * q_next * (1-done)
        # choose the loss function
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)
        critic_loss2 = criterion(q_value2, q_target)

        # optimize critics
        actor_net.zero_grad()
        critic_net1.zero_grad()
        critic_loss1.backward()
        critic_opt1.step()
        critic_net2.zero_grad()
        critic_loss2.backward()
        critic_opt2.step()

        ## update actor ##
        ## Delayed Policy Update -> Actor update fewer than Critic
        # actor loss
        ## TODO ##
        if epoch % self.policy_delay == 0:
            action = actor_net(state)
            # average all Q values from critic net. use (-) to minimize 
            actor_loss = - critic_net1(state, action).mean()

            # optimize actor
            actor_net.zero_grad()
            critic_net1.zero_grad()
            critic_net2.zero_grad()
            actor_loss.backward()
            actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            # target = tau * behavior + (1-tau) * target 
            target.data.copy_(tau * behavior.data + (1-tau) * target.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic1': self._target_critic_net1.state_dict(),
                    'target_critic2': self._target_critic_net2.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt1': self._critic_opt1.state_dict(),
                    'critic_opt2': self._critic_opt2.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net1.load_state_dict(model['critic1'])
        self._critic_net2.load_state_dict(model['critic2'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net1.load_state_dict(model['target_critic1'])
            self._target_critic_net2.load_state_dict(model['target_critic2'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt1.load_state_dict(model['critic_opt1'])
            self._critic_opt2.load_state_dict(model['critic_opt2'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(episode)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            # whether to draw the game scene
            if args.render:
                env.render()
            # select action, don't add noicse
            action = agent.select_action(state, False)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # move to next state and sum to total_reward
            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                # reward log
                rewards.append(total_reward)
                print(f"Run{n_episode} Reward = {total_reward:.4f}")
                break
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='td3.pth')
    parser.add_argument('--logdir', default='log/td3')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--noise_clip', default=.5, type=float)
    parser.add_argument('--action_limit', default=.2, type=float)
    parser.add_argument('--policy_delay', default=2, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    agent = TD3(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
