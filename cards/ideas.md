# Architecture
* [X] Use attention for hands and cards 
    * Already tried, but no success. However, I strongly believe that with right modelling of domain knowledge this should provide improvements to DQN performance (although state is simple)
* Use LSTM for summarizing previous game states to add notion of the game progress/history to the model

# Training
* [X] Train in stages
    * Initial opponent = random agent, then switch to self-play with soft update or use pretrained best model
* Use prioritzed replay buffer
* [X] Use target network as an opponent - simplify the training process 
* Play with hyperparameters
    * Start with lower epsilon (0.35) and decay it slower over larger num of episodes
    * Adapt loss/learning rates to lower (see DRON paper)

# Env
* Make rewards less sparse
    * Add reward for successful defence i.e.