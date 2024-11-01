
import time
import math
import torch
import numpy as np


class Data_Sampler(object):
	def __init__(self, data, device, reward_tune='no'):
		
		self.state = torch.from_numpy(data['observations']).float()
		self.action = torch.from_numpy(data['actions']).float()
		self.next_state = torch.from_numpy(data['next_observations']).float()
		reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
		self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

		self.size = self.state.shape[0]
		self.state_dim = self.state.shape[1]
		self.action_dim = self.action.shape[1]

		self.device = device

		if reward_tune == 'normalize':
			reward = (reward - reward.mean()) / reward.std()
		elif reward_tune == 'iql_antmaze':
			reward = reward - 1.0
		elif reward_tune == 'iql_locomotion':
			reward = iql_normalize(reward, self.not_done)
		elif reward_tune == 'cql_antmaze':
			reward = (reward - 0.5) * 4.0
		elif reward_tune == 'antmaze':
			reward = (reward - 0.25) * 2.0
		self.reward = reward

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))

		return (
			np.array(ind.cpu()),
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device),
		)

	def state_transform(self, state_n):
		state=np.array(self.state.cpu())
		next_state=np.array(self.next_state.cpu())
		s1min = (np.min(state, axis=0))
		s1max = (np.max(state, axis=0))
		s2min = (np.min(next_state, axis=0))
		s2max = (np.max(next_state, axis=0))
		smin = np.minimum(s1min, s2min)
		smax = np.maximum(s1max, s2max)
		Smax = []
		Smin = []
		j = []
		state_n = int(state_n)
		for i in range(self.state_dim):
			if smax[i] - smin[i] != 0:
				Smin.append(smin[i])
				Smax.append(smax[i])
			else:
				j.append(i)
		new_bservations = np.delete(state, j, axis=1)
		new_next_obs = np.delete(next_state, j, axis=1)
		s_change = ((state_n) * (new_bservations - Smin)) / (np.array(Smax) - Smin)
		s_next_change = ((state_n) * (new_next_obs - Smin)) / (np.array(Smax) - Smin)
		s_change = (s_change + 0.5) // 1
		s_next_change = (s_next_change + 0.5) // 1
		s_change_cur = 0
		s_change_next = 0
		s_change_cur1 = 0
		s_change_next1 = 0
		s_change_cur2=0
		s_change_next2=0
		q=0
		state_dim = new_bservations.shape[1]
		for i in range(state_dim):
			if q==0:
				s_change_cur = (np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur))
				s_change_next = (np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next))
				if s_change_cur.max() > 9 * 10 ** 17:
					q=1
			if q==1:
				s_change_cur1=(np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur1))
				s_change_next1=(np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next1))
				if s_change_cur1.max()>9 * 10 ** 17:
					q=2
			if q==2:
				s_change_cur2=(np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur2))
				s_change_next2=(np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next2))
		r = 1
		state_num = np.append(s_change_cur, s_change_next)
		state_num1=np.append(s_change_cur1,s_change_next1)
		state_num2=np.append(s_change_cur2,s_change_next2)
		print(state_num.max())
		print(state_num1.max())
		print(state_num2.max())
		state_num2_sort = state_num2 = np.zeros_like(state_num)
		order = np.argsort(state_num)
		order1=np.argsort(state_num1)
		state_num_sort = sorted(state_num)
		state_num_sort1=np.zeros_like(state_num)
		if q>0:
			state_num_sort1=sorted(state_num1)
		state_num_sort2=np.zeros_like(state_num)
		if q>1:
			state_num_sort2=sorted(state_num2)
		for i in range(2 * self.size - 1):
			j = i + 1
			if state_num_sort[j] == state_num_sort[i] and state_num_sort1[j]==state_num_sort1[i] and state_num_sort2[j]==state_num_sort2[i]:
				state_num2_sort[j] = state_num2_sort[i]
			else:
				state_num2_sort[j] = r
				r = r + 1
		print(state_num2_sort.max())
		state_num2_sort = sorted(state_num2_sort)
		for m in range(len(state_num)):
			state_num2[order[m]] = state_num2_sort[m]
		s_and_snext = np.split(state_num2, 2)
		print('all_state_num', len(state))
		print(state_num2.max())
		return  s_and_snext[0], s_and_snext[
			1], state_num2.max(), len(state)

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward