'''
Layout (cell numbers match the reference image):

    1  2  3
    4  5  6
    S  X  G
X is a hole and G is a destination
Let's go
'''
from __future__ import annotations
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
 

class GridWorld():
    def __init__(self):
        self.hole = 8
        self.start = 7
        self.goal = 9
        self.index_to_step = [[0,1],[0,-1],[1,0],[-1,0]] # Right,Left,Down,Up
        self.ACTION_NAMES = ("Right", "Left", "Down", "Up")

    def initialize(self):
        self.state = 7
        self.steps = 0
        return self.state
        
    def step(self,action : int) -> tuple:
        row, col = divmod(self.state - 1, 3)
        moves = self.index_to_step[action]
        row, col = np.clip(row + moves[0],min = 0,max = 2) , np.clip(col + moves[1],min = 0,max = 2)
        new_state = int(row * 3 + col + 1)

        self.state = new_state
        if(new_state == self.hole):
            reward = -10
        elif(new_state == self.goal):
            reward = 10
        else:
            reward = -1
        fall_to_hole = new_state == self.hole
        reached_goal = new_state == self.goal
        done = fall_to_hole or reached_goal

        return new_state,reward,reached_goal,done
    def render(self) -> None:
        symbols = [str(i) for i in range(1, 10)]
        symbols[self.hole - 1] = "X"
        symbols[self.goal - 1] = "G"
        symbols[self.state - 1] = (
            "A" if self.state != self.goal else "A/G"
        )
        for row in range(3):
            print(
                " | ".join(
                    f"{x:^3}"
                    for x in symbols[3 * row : 3 * row + 3]
                )
            )
        print()

class ActorCritic(nn.Module):
    def __init__(self,num_states = 9,num_actions = 4):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(num_states,64), # Input duoc one-hot-encoding !
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64,num_actions)
        self.critic = nn.Linear(64,1)
    def forward(self,states : torch.Tensor) -> tuple[torch.Tensor,torch.Tensor] :
        one_hot_state = nn.functional.one_hot(states.long() - 1,num_classes= 9)
        features = self.shared_net(one_hot_state.float())
        return self.actor(features), self.critic(features).squeeze(-1)
    
    def act(self,state : int) -> tuple[int,float,float]:
        tensor_state = torch.tensor([state])
        logits,Values = self(tensor_state)
        distribution = Categorical(logits = logits)
        action = distribution.sample()
        return action.item(),distribution.log_prob(action).item(),Values.item()

@dataclass
class PPOConfig:
    total_updates: int = 250
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.50


def compute_gae(
    rewards : torch.Tensor,
    dones : torch.Tensor,
    values : torch.Tensor,
    last_value : float,
    gamma : float ,
    gae_lambda : float ) -> tuple[torch.Tenso,torch.Tensor]:

    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = torch.tensor(last_value,dtype =torch.float32)
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + next_value * not_done * gamma -  values[t]
        gae = delta + not_done * gae * gae_lambda * gamma
        advantages[t] = gae
        next_value = values[t]
    return_values = advantages + values
    return advantages, return_values

def ppo_update(model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> None:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _ in range(cfg.update_epochs):
        indices = torch.randperm(len(states)) # For shuffling
        for start in range(0,len(states),cfg.minibatch_size):
            batch = indices[start : start + cfg.minibatch_size]
            logits,new_values = model(states[batch]) # (cfg.minibatch_size,4)
            distribution = Categorical(logits = logits) 
            new_log_probs = distribution.log_prob(actions[batch])

            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(
                ratio,1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages[batch]
            policy_loss = -torch.min(unclipped,clipped).mean()

            value_loss = 0.5 * (new_values - returns[batch]).pow(2).mean()

            entropy = distribution.entropy().mean() # High entropy means the policy is uncertain and explores more.

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg.max_grad_norm)
            optimizer.step()
            print(f"Policy loss: {policy_loss}")
            print(f"Value loss: {value_loss}")
            print(f"Entropy loss: {entropy}")


def train(seed : int = 42) -> ActorCritic:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1) # Check !

    cfg = PPOConfig()
    env = GridWorld()
    model = ActorCritic()
    optimizer = torch.optim.AdamW(model.parameters(),cfg.learning_rate)
    state = env.initialize()
    last_done = False
    recent_returns : list[float] = []
    episode_return = 0.0

    for update in range(cfg.total_updates):
        states, actions, log_probs = [], [], []
        rewards, dones, values = [], [], []
        for roll_step in range(cfg.rollout_steps):
            action,log_prob,value = model.act(state)
            new_state,reward,_,done = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            state = new_state
            episode_return += reward
            last_done = done
            if done:
                recent_returns.append(episode_return)
                recent_returns = recent_returns[-100:]
                episode_return = 0
                state = env.initialize()
        
        states_t = torch.tensor(states, dtype=torch.long)
        actions_t = torch.tensor(actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        old_values_t = torch.tensor(values, dtype=torch.float32)

        # If after cfg.rollout_steps, the trajectory isn't done,we calculate the value at that state.
        with torch.no_grad():
            _, end_value = model(torch.tensor([state]))
        last_value = 0.0 if last_done else end_value.item()

        advantages,returns = compute_gae(rewards_t,dones_t,old_values_t,last_value,cfg.gamma,cfg.gae_lambda)
        print(f"EPOCHS: {update}")
        ppo_update(model,optimizer,states_t,actions_t,old_log_probs_t,old_values_t,advantages,returns,cfg)

    return model

def show_greedy_episode(model: ActorCritic) -> None:
    env = GridWorld()
    state = env.initialize()
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            logits,value = model(torch.tensor([state]))
        probas = torch.softmax(logits,dim = -1).squeeze(0)
        action = torch.argmax(probas).item()
        print(
            f"cell={state}, V(s)={value.item():.3f}, "
            f"action={env.ACTION_NAMES[action]}",
            f"probabilities={probas.numpy().round(3)}"
        )
        state,reward,reached_goal,done = env.step(action)
        total_reward += reward
        env.render()
    print(f"Reached goal: {reached_goal} | total reward: {total_reward:.3f}")



    

def evaluate(model: ActorCritic, episodes: int = 100) -> tuple[float, float]:
    """Evaluate the deterministic (argmax) policy."""
    successes = 0
    episode_lengths = []

    for _ in range(episodes):
        env = GridWorld()
        state = env.initialize()
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                logits, _ = model(torch.tensor([state]))
            action = logits.argmax(dim=-1).item()
            state, _, reached_goal,done = env.step(action)
            steps += 1

        successes += int(reached_goal)
        episode_lengths.append(steps)

    return successes / episodes, float(np.mean(episode_lengths))

if __name__ == "__main__":
    trained_model = train()
    success_rate, mean_steps = evaluate(trained_model)
    print(f"\nEvaluation: success_rate={success_rate:.1%}, mean_steps={mean_steps:.2f}")
    show_greedy_episode(trained_model)

        

