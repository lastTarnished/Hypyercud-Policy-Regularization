import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim,scale,shift, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.size = self.state.shape[0]
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.scale = scale
		self.shift = shift


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		transition= [
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			np.array(ind),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		]

		transition[3]=self.scale * transition[3] + self.shift

		return transition


	def convert_D4RL(self, dataset, reward_tune='no'):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		reward=dataset['rewards'].reshape(-1,1)
		self.reward = reward
		#self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	def state_transform(self, state_n):
		state=np.array(self.state)
		next_state=np.array(self.next_state)
		print('self.state',self.state)
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
		s_change_cur2 = 0
		s_change_next2 = 0
		q = 0
		state_dim = new_bservations.shape[1]
		for i in range(state_dim):
			if q == 0:
				s_change_cur = (np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur))
				s_change_next = (np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next))
				if s_change_cur.max() > 9 * 10 ** 17:
					q = 1
			if q == 1:
				s_change_cur1 = (np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur1))
				s_change_next1 = (np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next1))
				if s_change_cur1.max() > 9 * 10 ** 17:
					q = 2

			if q == 2:
				s_change_cur2 = (np.int64(s_change[:, i]) + np.int64((state_n + 1) * s_change_cur2))
				s_change_next2 = (np.int64(s_next_change[:, i]) + np.int64((state_n + 1) * s_change_next2))
		r = 1
		state_num = np.append(s_change_cur, s_change_next)
		state_num1 = np.append(s_change_cur1, s_change_next1)
		state_num2 = np.append(s_change_cur2, s_change_next2)
		print(state_num.max())
		print(state_num1.max())
		print(state_num2.max())
		state_num2_sort = state_num2 = np.zeros_like(state_num)
		order = np.argsort(state_num)
		#order1 = np.argsort(state_num1)
		state_num_sort = sorted(state_num)
		state_num_sort1 = np.zeros_like(state_num)
		if q > 0:
			state_num_sort1 = sorted(state_num1)
		state_num_sort2 = np.zeros_like(state_num)
		if q > 1:
			state_num_sort2 = sorted(state_num2)
		for i in range(2 * self.size - 1):
			j = i + 1
			if state_num_sort[j] == state_num_sort[i] and state_num_sort1[j] == state_num_sort1[i] and state_num_sort2[
				j] == state_num_sort2[i]:
				state_num2_sort[j] = state_num2_sort[i]
			else:
				state_num2_sort[j] = r
				r = r + 1
		state_num2_sort = sorted(state_num2_sort)
		for m in range(len(state_num)):
			state_num2[order[m]] = state_num2_sort[m]
		s_and_snext = np.split(state_num2, 2)
		print('all_state_num', len((state)))
		print(state_num2.max())
		return s_and_snext[0], s_and_snext[1], state_num2.max(), len(state)