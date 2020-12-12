import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import random


class Storage():

    def __init__(self, obs_shape, num_steps, device):
        self.current_state_batch = torch.zeros(num_steps,*obs_shape)
        self.reward_batch = torch.zeros(num_steps)
        self.action_batch = torch.zeros(num_steps)
        self.next_state_batch = torch.zeros(num_steps, *obs_shape)
        self.done_batch = torch.zeros(num_steps)
        self.step = 0
        self.num_steps = num_steps
        self.device = device
        self.flag = 0

    def store(self, current_state, action, reward, next_state, done):
        """
        Store the given SARS transition objective
        :return:
        """
        self.current_state_batch[self.step] = current_state.clone().detach()
        self.action_batch[self.step] = action.clone().detach()
        self.reward_batch[self.step] = torch.tensor(reward)
        self.next_state_batch[self.step] = next_state.clone().detach()
        self.done_batch[self.step] = torch.tensor(done).clone().detach()
        if (self.step % (self.num_steps-1) == 0) and (self.step != 0) :
            self.flag = 1
            self.step = 0
        else:
            self.step += 1
        return self.step


    def batch_generator(self, mini_batch_size):
        """
        Random sampler.
        Must do : Prioritize experience replay
        :return:
        """
        sampler = BatchSampler(SubsetRandomSampler(range(self.step)), mini_batch_size, drop_last = True)

        for indices in sampler:

            current_state_batch = torch.FloatTensor(self.current_state_batch)[indices].to(self.device)
            action_batch = self.action_batch[indices].to(self.device)
            reward_batch = torch.FloatTensor(self.reward_batch)[indices].to(self.device)
            next_state_batch = torch.FloatTensor(self.next_state_batch)[indices].to(self.device)
            done_batch = self.done_batch[indices].to(self.device)
            yield current_state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def sample(self, mini_batch_size):
        batch_size = 0
        if self.flag == 0:
            batch_size = self.step
        else:
            batch_size = self.num_steps
        num_steps_list = [i for i in range(batch_size)]
        #num_steps_list = [i for i in range(100)]
        indices = random.sample(num_steps_list , mini_batch_size)

        current_state_batch = torch.FloatTensor(self.current_state_batch)[indices].to(self.device)
        action_batch = torch.tensor(self.action_batch[indices], dtype = torch.int64).to(self.device)
        reward_batch = torch.FloatTensor(self.reward_batch)[indices].to(self.device)
        next_state_batch = torch.FloatTensor(self.next_state_batch)[indices].to(self.device)
        done_batch = self.done_batch[indices].to(self.device)
        return current_state_batch, action_batch, reward_batch, next_state_batch, done_batch



