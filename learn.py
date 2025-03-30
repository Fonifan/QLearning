import random
import gymnasium as gym
import minigrid

from q import q_learning

def policy(observation):
   return int(input("Enter action: "))


env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
q_learning(env, env.observation_space, env.action_space)

print(env.action_space)
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()