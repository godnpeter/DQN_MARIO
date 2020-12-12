import model
import torch.nn.init as init
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import torch
import utils
import copy
import time

"""
https://curt-park.github.io/2018-05-17/dqn/ ==> hyperparameter list
https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c ==> reference
"""

class Agent:

    def __init__(self, env, env_name,storage, explore, lmbda, num_actions, device, explore_timesteps, writer):
        self.env_name = env_name
        self.action_space = env.action_space
        self.train_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        self.target_net = model.DQN(env.observation_space.shape,env.action_space.n).to(device)
        self.storage = storage
        self.env = env
        self.init_explore = 1
        self.final_explore = explore
        self.explore_timesteps = explore_timesteps
        self.lmbda = lmbda
        self.device = device
        self.writer = writer
        self.loss_array = []

        # initialize target_net with train_net parameters
        self.target_net.load_state_dict(self.train_net.state_dict())

        # Always, only the train_net should be trained
        self.train_net.train()
        self.target_net.eval()

        #The Huber loss
        self.loss_func = nn.SmoothL1Loss()

        #TODO : implement anneealing learning rate
        self.optimizer = optim.Adam(self.train_net.parameters(), lr = 0.00025)

        # Attributes to save the best model
        self.best_reward_mean = None
        self.best_model = None

    def action(self, observation):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype = torch.float32).to(self.device)
            action_scores = self.train_net(observation.unsqueeze(0)).cpu()
            action = np.argmax(action_scores)
        return action

    def train(self, num_timesteps):

        mean_reward_list = []
        tot_reward = 0
        current_state = self.env.reset()
        current_state = torch.FloatTensor(current_state)
        num_episodes = 0
        done = False
        for timestep in range(num_timesteps):
            self.env.render()

            #epsilon = utils.linear_explore(self.init_explore, self.final_explore, self.explore_timesteps, timestep)
            epsilon = utils.update_epsilon(timestep)
            self.writer.add_scalar('num_episodes/epsilon', epsilon, timestep)
            if random.random() > epsilon:
                action = self.action(current_state)
            else:
                action = torch.tensor(self.env.action_space.sample())

            # the observation returned by the environment is a LazyFrame object, which I don't really want
            # convert this next_state into a numpy array. Much more easier to handle
            #Todo
            # action.item()은 마리오를 위한것
            # 아타리를 할꺼면 action 만 집어넣자
            next_state, reward, done, info = self.env.step(action.item())

            tot_reward += reward
            next_state = torch.FloatTensor(next_state)

            size = self.storage.store(current_state, action,  reward, next_state, done)

            # 초반에는 어느정도 experience를 버퍼에 쌓아두고나서 학습을 하고싶어서 이렇게한다.

            #if timestep > 30000:

            if timestep > 30000:
                if timestep == 30001:
                    print('Training Starts : ', time.time())

                loss = self.optimize(timestep)
                self.writer.add_scalar('timestep/loss', loss, timestep)

                if timestep % 2000 == 0:
                    print('timestep : ', timestep)
                    print('Loss : ', loss)

            if done:
                mean_reward_list.append(tot_reward)
                mean_reward = np.mean(mean_reward_list[-100:])

                if self.best_reward_mean is None or mean_reward > self.best_reward_mean:
                    self.best_reward_mean = mean_reward

                print("%d:  %d games, mean reward %.3f, current reward %.3f ,(epsilon %.2f)" % (timestep, len(mean_reward_list), mean_reward, tot_reward,epsilon))

                current_state = self.env.reset()
                current_state = torch.FloatTensor(current_state)
                self.writer.add_scalar('num_episodes/reward', tot_reward, timestep)
                num_episodes += 1
                tot_reward = 0
            else:
                current_state = next_state

    def optimize(self, timestep):
        """
        We want to optimize the train_net everytime, but want to optimize the target_net every N times.
        Actually, the target_net should only copy the parameters of the train_net, not be optimized.
        torch has the function


        """
        update_step = 5000
        current_state_batch = None
        reward_batch = None
        next_state_batch = None
        done_batch = None

        # 이거 이상하게 보니까 학습이 아예 진행이 안되는데...?!?!!?!?!?!?!?!?!?!?!?!!?!?!!?!?
        # Check whether model parameter has been updated.... And it has!
        # old = list(self.train_net.parameters())[0].clone()
        # print("="*50)
        # print(self.train_net.state_dict().get('cnn1.weight')[0][3])
        # print("="*50)
        # print(self.target_net.state_dict().get('fc1.weight'))

        loss_array = []
        # Obtain the train_batch
        batch = self.storage.sample(32)

        current_state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        # This is where Double-DQN jumps in
        next_action_index = self.train_net(next_state_batch).detach().max(1)[1]
        target_batch = reward_batch + self.lmbda * self.target_net(next_state_batch).gather(1, next_action_index.unsqueeze(-1)).squeeze() * (1-done_batch)


        #target_batch = reward_batch + self.lmbda * self.target_net(next_state_batch).max(1)[0].detach() * (1 - done_batch)

        #train_batch = self.train_net(current_state_batch)
        #train_batch = train_batch[torch.arange(train_batch.size(0)), action_batch]
        train_batch = self.train_net(current_state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)


        loss = self.loss_func(train_batch, target_batch)
        self.optimizer.zero_grad()
        loss.backward()
        # Not sure about this... Is this the main problem? ==> Doesn't seems so...
        for param in self.train_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if timestep % update_step == 0:
            # Successfully checked target network parameters being udpated
            print('NETWORK UPDATED AND SAVED')
            self.target_net.load_state_dict(self.train_net.state_dict())
            torch.save(self.train_net.state_dict(), self.env_name + '-best_model_')


        # Check whether the train_net parameters has been updated.... And it has!!
        # new = list(self.train_net.parameters())[0].clone()

        loss_array.append(loss)

        return (np.sum(np.array(loss_array)))



    def evaluate(self):
        """
        used to play the best learned model
        """
        self.train_net.load_state_dict(torch.load(self.env_name + '-best_model_'))

        mean_reward_list = []
        tot_reward = 0
        current_state = self.env.reset()
        current_state = torch.FloatTensor(current_state)
        num_episodes = 0
        timestep = 0
        while True:
            self.env.render()
            #if random.random() > 0.1:
            action = self.action(current_state)
            #else:
            #    action = torch.tensor(self.env.action_space.sample())

            # the observation returned by the environment is a LazyFrame object, which I don't really want
            # convert this next_state into a numpy array. Much more easier to handle
            next_state, reward, done, info = self.env.step(action.item())
            tot_reward += reward
            next_state = torch.FloatTensor(next_state)
            timestep +=1

            if done:
                mean_reward_list.append(tot_reward)
                mean_reward = np.mean(mean_reward_list[-100:])

                print("%d:  %d games, mean reward %.3f" % (timestep, len(mean_reward_list), mean_reward))

                current_state = self.env.reset()
                current_state = torch.FloatTensor(current_state)
                num_episodes += 1
                tot_reward = 0
            else:
                current_state = next_state


