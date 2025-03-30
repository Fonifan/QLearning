# Overall notes
* Struggling to find good state representation (DQN comes in handy)
* The book algorithm defines a fully known state at all times, the actual env is based on agent visibility (so agent mightn't know the end state location at initial points). Book recommends to set Q(s,a) to non-zero except for terminal, but have to digress because of the envs constraints

# Q
* Was struggling with state representation, went for string (hashes) of everything available in state
* Algorithm was struggling with finding the most optimal path initially, capping the learning at ~0.97 reward and ~100 steps to reach target. Which led to an unoptimal path
    * Tried doing multistage Q estimation, where second stage would have baseline reward of 1 (not to mess up the learned Q) and decreased max_steps - Failed. The Q overwriting started and messed up learned paths
    * Saw the trick (in DQN tutorial) with epsilon decay - model finds the most optimal path (~29 steps) in ~100 episodes.

# DQN
* Updated initial 2 FC layers to Conv (for image) + 2 FC (direction gets injected into first)
* Still struggling with wasted moves in test environment (test env - human visible rendering of the grid)
    * However the training converges rather quickly:
        ```
        Episode 108 finished after 121 steps with reward 0.89365234375
        Episode 109 finished after 175 steps with reward 0.84619140625
        Episode 110 finished after 120 steps with reward 0.89453125
        Episode 111 finished after 117 steps with reward 0.89716796875
        Episode 112 finished after 115 steps with reward 0.89892578125
        Episode 113 finished after 121 steps with reward 0.89365234375
        Episode 114 finished after 233 steps with reward 0.79521484375
        Episode 115 finished after 226 steps with reward 0.8013671875
        Episode 116 finished after 322 steps with reward 0.7169921875
        Episode 117 finished after 309 steps with reward 0.72841796875
        Episode 118 finished after 435 steps with reward 0.61767578125
        Episode 119 finished after 89 steps with reward 0.92177734375
        Episode 120 finished after 116 steps with reward 0.898046875
        Episode 121 finished after 148 steps with reward 0.869921875
        Episode 122 finished after 311 steps with reward 0.7266601562499999
        Episode 123 finished after 145 steps with reward 0.87255859375
        Episode 124 finished after 104 steps with reward 0.90859375
        Episode 125 finished after 345 steps with reward 0.69677734375
        Episode 126 finished after 107 steps with reward 0.90595703125
        Episode 127 finished after 500 steps with reward 0.560546875
        ```
* Made direction a one-hot vector
    * Much better convergence (episodes 200+ already are at ~50 steps average)

* Problem with testing. The shortest path is learned during training, but due to starting direction randomness, during testing agent starts in an unseen state and gets stuck.
    * Fixed with setting random_seed for both training and testing