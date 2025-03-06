#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from agent import Agent
from dqn_model import DuelingDQN
"""
you can import any package and define any extra function as you need
"""


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, beta):
        assert len(self.priorities) == len(self.buffer), "Priorities and buffer size mismatch"
        probabilities = np.clip(np.array(self.priorities), 1e-5, None) / max(sum(self.priorities), 1e-5)
        assert np.all(np.isfinite(probabilities)), "Non-finite probabilities detected"
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, torch.tensor(weights, dtype=torch.float32)


    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.flatten()
        assert len(td_errors) == len(indices), "Mismatch between td_errors and indices lengths"
        for idx, td_error in zip(indices, td_errors):
            td_error = float(td_error)
            assert np.isfinite(td_error), f"Non-finite TD Error encountered: {td_error}"
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha



#gamma=0.99, lr=1e-4, tau=1e-3, buffer_size=100000, min_buffer_size=50000, batch_size=64, alpha=0.6, beta=0.4
class Agent_DQN(Agent):
    
    def __init__(self, env, args):
        super(Agent_DQN, self).__init__(env)
        self.env = env
        self.gamma = 0.99
        self.tau = 5e-5
        self.batch_size = 16
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.min_buffer_size = 40000
        self.lr = 5e-5
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(100000, self.alpha)

        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingDQN(in_channels=4, num_actions=env.action_space.n).to(self.device)
        self.target_network = DuelingDQN(in_channels=4, num_actions=env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), self.lr)

        # Sync weights
        self.update_target_network(1.0)

        # Epsilon-greedy policy
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def update_target_network(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
        
    def step(self, state, action, reward, next_state, done):
        reward = float(reward)  # Ensure reward is a float for computation

        with torch.no_grad():
            # Convert to tensors and permute for channels-first format
            done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)

            # Calculate best actions for the next state
            best_actions = self.q_network(next_state_tensor).argmax(1, keepdim=True)

            # Compute TD error
            td_error = reward + self.gamma * self.target_network(next_state_tensor).gather(1, best_actions) * (1 - done_tensor) - self.q_network(state_tensor)[0, action]
            assert np.isfinite(td_error.item()), f"TD Error is not finite: {td_error.item()}"


        # Add experience to replay buffer
        self.memory.add((state, action, reward, next_state, done), td_error.item())

        # Start learning only if the buffer is sufficiently filled
        if len(self.memory.buffer) >= self.min_buffer_size:
            self.learn()




    def learn(self):
        experiences, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors and ensure proper shape
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        weights = weights.to(self.device).unsqueeze(-1)

        # Compute Q targets
        best_actions = self.q_network(next_states).argmax(1, keepdim=True)
        q_targets_next = self.target_network(next_states).gather(1, best_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute Q expected
        q_expected = self.q_network(states).gather(1, actions)

        # Compute TD error
        td_errors = q_targets - q_expected
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Soft update the target network
        self.update_target_network(self.tau)


    def train(self, episodes=100000, max_steps=3000):
            best_reward = -float("inf")  # Initialize the best reward to a very low value
            frame_count = 0  # Track total frames
            rewards_window = deque(maxlen=1000)  # Track rewards for last 1000 episodes
            episode_rewards = []  # List to store rewards for each episode

            try:
                for episode in range(1, episodes + 1):
                    state = self.env.reset()
                    total_reward = 0
                    for step in range(max_steps):
                        frame_count += 1  # Increment frame count
                        action = self.act(state)
                        result = self.env.step(action)
                        if len(result) == 4:
                            next_state, reward, done, info = result
                        else:
                            next_state, reward, done, info, *_ = result
                        self.step(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                        if done:
                            break

                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    rewards_window.append(total_reward)  # Add reward to the window
                    avg_reward = np.mean(rewards_window)  # Calculate average reward
                    episode_rewards.append(total_reward)  # Append episode reward to the list

                    # Print episode details including epsilon
                    print(f"Episode {episode}, Total Reward: {total_reward}, "
                        f"Avg Reward (last 1000): {avg_reward:.2f}, "
                        f"Frame: {frame_count}, Epsilon: {self.epsilon:.4f}")

                    # Save model if it achieves a new best reward
                    if total_reward > best_reward:
                        best_reward = total_reward
                        torch.save(self.q_network.state_dict(), "best_model.pth")
                        print(f"New best model saved with reward: {best_reward}")

            except KeyboardInterrupt:
                print("Training interrupted! Generating learning curve...")
            finally:
                # Generate and save learning curve
                self.plot_learning_curve(episode_rewards)

    def plot_learning_curve(self, episode_rewards):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label="Episode Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Learning Curve")
        plt.grid()
        plt.legend()
        plt.savefig("learning_curve.png")
        plt.close()
        print("Learning curve saved as 'learning_curve.png'")



    def make_action(self, state, test=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        return action

    def init_game_setting(self):
        # If you have any special settings to initialize, do it here
        # For example, reset the epsilon value if needed
        #self.epsilon = self.epsilon_min  # If you want to fix epsilon during testing
        if os.path.exists("best_model.pth"):
            self.q_network.load_state_dict(torch.load("best_model.pth", map_location=self.device))
           
        pass