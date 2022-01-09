import torch


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
    
    def sample_tr_st_r(self, batch_size):
        # yield transitions (st+a), next states and rewards per batch as seperate matrices

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            batch = star_copy[k:k+batch_size]
            yield (batch[:, :self.n_obs+self.n_act], batch[:, self.n_obs+self.n_act : 2*self.n_obs+self.n_act], batch[:,-1:])
    
    def sample_states(self, batch_size):
        # yield states per batch

        star_copy = torch.clone(self.get_stars())

        #shuffle the tensor
        idx = torch.randperm(self.length)
        star_copy = star_copy[idx].view(star_copy.size())

        for k in range(0, self.length, batch_size):
            yield star_copy[k:k+batch_size, :self.n_obs]