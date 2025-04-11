import matplotlib.pyplot as plt
import numpy as np

epsilon0 = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000

epsilons = []
epsilon = epsilon0

for episode in range(episodes):
    epsilons.append(epsilon)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

plt.figure(figsize=(8, 4))
plt.plot(np.arange(episodes), epsilons, label='Epsilon')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exponential Decay of Epsilon')
plt.legend()
plt.grid(True)
plt.show()