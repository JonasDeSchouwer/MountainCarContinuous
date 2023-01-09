from __future__ import annotations

import torch
import logging
import pickle


# device
if torch.cuda.is_available():
    print("running on GPU")
    DEVICE = torch.device("cuda")
else:
    print("running on CPU")
    DEVICE = torch.device("cpu")


def nthroot(a, n):
    assert a >= 0, "you don't want to take the nth root of a negative number, pal"
    return a ** (1/n)


def soft_update(target: torch.nn.Module, approx: torch.nn.Module, tau):
    """
    returns a new state_dict for the target according to the formula
    tau * approx + (1-tau) * target
    """
    target_sd = target.state_dict()
    approx_sd = approx.state_dict()

    for key in approx_sd:
        target_sd[key] = approx_sd[key]*tau + target_sd[key]*(1-tau)
    
    target.load_state_dict(target_sd)   # type: ignore


class Memory:
    # saves state-action-next_state-reward (star) as a Nx. matrix: first n_obs columns (obs), next n_act columns (act), next n_obs columns(next_obs), last column (rew)
    # state = observations
    # transition = state + action

    def __init__(self, num_observations, num_actions, capacity):
        self.n_obs = num_observations
        self.n_act = num_actions
        self.star = torch.zeros((capacity, self.n_obs + self.n_act + self.n_obs + 1), device=DEVICE)
        self.length = 0
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            return self.star[self.length-index]
        else:
            return self.star[index]
    

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> Memory:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_stars(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_stars(), f)
    
    def load_stars(self, path, num_stars):
        with open(path, 'rb') as f:
            self.add_stars(pickle.load(f)[:num_stars])


    def get_stars(self):
        return self.star[:self.length]

    def get_states(self):
        return self.star[:self.length, :self.n_obs]
    
    def get_actions(self):
        return self.star[:self.length, self.n_obs:self.n_obs+self.n_act]
    
    def get_next_states(self):
        return self.star[:self.length, self.n_obs+self.n_act : 2*self.n_obs+self.n_act]
    
    def get_rewards(self):
        return self.star[:self.length, -1]

    def add_state(self, state, action, next_state, expected_reward=0):
        self.star[self.length][:self.n_obs] = torch.as_tensor(state, device=DEVICE)
        self.star[self.length][self.n_obs:self.n_obs+self.n_act] = torch.as_tensor(action, device=DEVICE)
        self.star[self.length][self.n_obs+self.n_act : 2*self.n_obs+self.n_act] = torch.as_tensor(next_state, device=DEVICE)
        self.star[self.length][-1] = expected_reward
        self.length += 1

    def add_star(self, star):
        self.star[self.length] = torch.as_tensor(star, device=DEVICE)
        self.length += 1
    
    def add_stars(self, stars):
        self.star[self.length : self.length+len(stars)] = torch.as_tensor(stars, device=DEVICE)
        self.length += len(stars)
    
    def remove_first_states(self, n):
        self.star = torch.roll(self.star, shifts=-n, dims=0)
        self.star[-n:] = 0
        self.length -= n

    def sample_st_a_r(self, batch_size):
        # yield states, actions and rewards per batch as seperate matrices

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs], batch[:, self.n_obs:self.n_obs+self.n_act], batch[:,-1:])
        
    def sample_tr_r(self, batch_size):
        # yield transitions (st+a) and rewards per batch as seperate matrices
        
        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs+self.n_act], batch[:,-1:])

    def sample_st_a_st_r(self, batch_size):
        # yield states, actions, next states and rewards per batch as seperate matrices

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs], batch[:, self.n_obs:self.n_obs+self.n_act], batch[:, self.n_obs+self.n_act : 2*self.n_obs+self.n_act], batch[:,-1:])
    
    def sample_tr_st_r(self, batch_size, max_batches=None):
        # yield transitions (st+a), next states and rewards per batch as seperate matrices

        if max_batches == None:
            max_batches = self.length//batch_size + 1

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, min(self.length, max_batches*batch_size), batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs+self.n_act], batch[:, self.n_obs+self.n_act : 2*self.n_obs+self.n_act], batch[:,-1:])
    
    def sample_states(self, batch_size, max_batches=None):
        # yield states per batch

        if max_batches == None:
            max_batches = self.length//batch_size + 1

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, min(self.length, max_batches*batch_size), batch_size):
            yield star_copy[k:k+batch_size, :self.n_obs]


class OrnsteinUhlenbeck:
    """
    models an Ornstein-Uhlenbeck process with parameters sigma and theta.
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    this is used to introduce temporal coherence (and mean-reverting) to the noise of our policy network
    """

    def __init__(self, theta, sigma, shape=None, init=None):
        self.theta = theta
        self.sigma = sigma

        if init is None:
            if shape is None:
                self.value = torch.zeros(size=(), device=DEVICE)
                self.shape = tuple()
                logging.info("init and shape are both None: default value 0")
            else:
                self.value = torch.zeros(size=shape, device=DEVICE)
                self.shape = shape
        else:
            if shape is not None:
                assert init.shape == shape, "keywords shape and init do not match"
            self.value = init
            self.shape = init.shape
    
    def step(self):
        self.value = (1-self.theta)*self.value + torch.normal(0, std=self.sigma, size=self.shape, device=DEVICE)
        return self.value


class GaussianNoise:
    """
    returns Gaussian noise at each step, with a given mean and std
    """

    def __init__(self, mean=0, std=1, shape=None):
        self.mean = mean
        self.std = std

        if shape is None:
            self.value = torch.zeros(size=(), device=DEVICE)
            self.shape = tuple()
            logging.info("init and shape are both None: default value 0")
        else:
            self.value = torch.zeros(size=shape, device=DEVICE)
            self.shape = shape
    
    def step(self):
        return torch.normal(0, std=self.std, size=self.shape, device=DEVICE)