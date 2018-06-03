Project to implement the DQN reinforcement in a similar fashion to that found in "Playing Atari with Deep Reinforcement Learning" (www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

The agent is applied to CarRacing-V0 environment from Open AI gym.

The structure of the q-network is a bit different from the original paper; it has been made smaller to facilitate training, especially if training is performed on a CPU.

Checkpoints can be found under the data folder. checkpoint_Orig corresponds to the training done with parameters matching that of the original paper. Checkpoint02 corresponds to the training with some parameters changed. Checkpoint_orig currently has the best set of weights. 

The current code allows training from scratch or loading in an existing checkpoint. Continuing from a checkpoint allows the agent to run or continue to be trained.

Requirements:
- Python 3.5
- OpenAI Gym (with Box2D)
- Tensorflow
- numpy
- scikit-image
