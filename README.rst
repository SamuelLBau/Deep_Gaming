This project will attempt to implement a deep Q algorithm to learn how to play various games. The initial goal is to learn to play the game of Snake. Once this has been implemented, we may try the framework on other simple games such as asteroids.

This project is primarily built using tensorflow.

The code base can be cloned using the following command:
    git clone https://github.com/SamuelLBau/Deep_Gaming.git

The python environment for this code base can by prepared by running setup.sh, or setup_GPU.sh if a GPU is available.
If running on a windows machine, the script should also work as a .cmd file, or each command run manually

Setup assumes anaconda has been installed:

setup.sh and setup_gpu.sh approximatly runs the following commands

conda create --name Deep_Gaming python=3.5.5
source activate Deep_Gaming
pip install numpy
pip install matplotlib
pip install imageio
pip install tensorflow #(tensorflow-gpu if GPU is available)
pip install gym
pip install cmake #Required for gym[atari]
pip install gym[atari]
conda install -c https://conda.anaconda.org/kne pybox2d #Only needed for CarRacing, can be ignored if causes problems
echo "Please run <source activate" ${virtual_env} ">"

The virtual environment can then be loaded using <source activate Deep_Gaming> or <source activate Deep_Gaming_GPU>



To run examples:

python load_sample_networks.py:
    #This will load networks into the work area (saved_networks)
    #If this will cause existing networks to be overwritten, confirmation will be required
    
python run_test_example.py [--env <env_name>]:
    #By default, this will run the MsPacman example, as a decent sample network has been provided
    
    #You can choose to run a different environment by adding a --env <environment_name> flag
    
    #Supported environments are: snake,MsPacman-v0,Asteroids-v0,CarRacing-v0
    
    #NOTE: We do not have a method of displaying graphics from the server. Results will be saved to a .gif file
    #Which can be loaded to another computer to play
    #Also NOTE: CarRacing-v0 in particular only works when rendering is enabled, so it will not run on the server
    
python run_training_example.py [--env <env_name>]:
    #By default, this will run train MsPacman, as a decent sample network has been provided
    #network
    
    #You can choose to run a different environment by adding a --env <environment_name> flag
    #Supported environments are: snake,MsPacman-v0,Asteroids-v0,CarRacing-v0

To Graph results:
    
python generate_graphs.py --dir <dir_path>
    #This function will grab the .rewards and .qs files from the specified directory and plot the results
    
    #If the --save_rewards flag is set during training, these files will be generated in appropriate saved_networks directory
    #An example would be python generate_graphs.py --dir ./saved_networks/MsPacman-v0_PacNet
    
Using the tool:
    #The main 
    
    

Key references:

Blog page about deep-Q learning: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html

Longer paper about Deep-Q learning: http://proceedings.mlr.press/v48/gu16.pdf

Good source for convolutional Neural Nets: http://cs231n.github.io/convolutional-networks/#conv

AI_Gym (May allow for more complex games): https://gym.openai.com/

Arcade_Learning_environment (Specific to Atari games): https://github.com/mgbellemare/Arcade-Learning-Environment

Git repository for a3c using pytorch (not Deep-Q): https://github.com/ikostrikov/pytorch-a3c
