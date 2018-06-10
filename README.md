This project will attempt to implement a deep Q algorithm to learn how to play various games. The initial goal is to learn to play the game of Snake. Once this has been implemented, we may try the framework on other simple games such as asteroids.

This project should be built using tensorflow, and possibly pytorch.

This project was install using anaconda

conda create --name Deep_Gaming python=3.5.5
pip install numpy
pip install matplotlib
pip install tensorflow #(tensorflow-gpu if GPU is available)
pip install gym
pip install cmake #Required for gym[atari]
pip install gym[atari]
pip install Box2D #Only needed for CarRacing, can be ignored if causes problems


Key references:

Blog page about deep-Q learning: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html

Longer paper about Deep-Q learning: http://proceedings.mlr.press/v48/gu16.pdf

Good source for convolutional Neural Nets: http://cs231n.github.io/convolutional-networks/#conv

AI_Gym (May allow for more complex games): https://gym.openai.com/

Arcade_Learning_environment (Specific to Atari games): https://github.com/mgbellemare/Arcade-Learning-Environment

Git repository for a3c using pytorch (not Deep-Q): https://github.com/ikostrikov/pytorch-a3c
