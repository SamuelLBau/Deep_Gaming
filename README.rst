This project will attempt to implement a deep Q algorithm to learn how to play various games. The initial goal is to learn to play the game of Snake. Once this has been implemented, we may try the framework on other simple games such as asteroids.

This project is primarily built using tensorflow.

**===============================================================================**
                                **Getting Code**
**===============================================================================**
The code base can be cloned using the following command:
    git clone https://github.com/SamuelLBau/Deep_Gaming.git
    

**===============================================================================**
                              **Code Organization**
**===============================================================================**

Deep_Gaming/
  doc/               # Miscellaneous Reports, proposals and example results
    example_results/ #.gif files of each game being run by network
    
    final_report/    #All Latex files used to generate final report .pdf
  src/               #All code for the project
    * Described below
    
  .gitignore
  
  README.rst
  
  setup.sh           #Prepares Anaconda Virtual Environment for project, tensorflow for CPU
  
  setup_GPU.sh       #Prepares Anaconda Virtual Environment for project, tensorflow for GPU

Deep_Gaming/src
  cfn/               #Contains all network description files (.prototxt files)
  
  sample_networks/   #Contains sample networks. Do not edit these files when testing.
  
  saved_networks/    #Contains all networks trained and saved by system. (Generated during run)
  
  caffe_builder.py        #Builds network from description file
  
  caffe_builder_utils.py  #Utility functions for above
  
  deepQ.py                #Main file for learning, can be used without the run_*_demo.py files
  
  load_sample_networks.py #Loads sample networks from sample_networks/ to /saved_networks. Use to reset space
  
  generate_graphs.py      #Used to generate graphs of score and Q sum over various games
  
  run_test_demo.py     #Used to configure and run pre-trained networks to generate demo episodes
  
  run_test_demo.ipynb  #Same as above, but as a notebook. This has more or less the same functionality, but is able to display the final frame of the gameplay.
  .gif files are still generated. We have been unable to test this on the server.
  
  run_training_demo.py #Used to configure and train networks on selected environments
  
  run_training_demo.ipynb #Same as above, but as a notebook. We have been unable to test this on the server. 
  The training has no display that is of any interest, we recommend using the .py file as it has cleaner and more
  descriptive print statements.
  
  snake_game.py           #Simple snake environment, can be used in place of gym environments
  
**===============================================================================**
                                **Setting up**
**===============================================================================**

NOTE: If running jupyter, please use launch-py3torch-gpu.sh, if just using python, any are find, but gpu preferred

The python environment for this code base can by prepared by running setup.sh, or setup_GPU.sh if a GPU is available.
If running on a windows machine, the script should also work as a .cmd file, or each command can be run manually

Setup assumes anaconda has been installed and properly configured,
Only modules that are not included with a default anaconda install are included 

setup.sh and setup_gpu.sh approximatly runs the following commands
setup_jupyter.sh does not use Anaconda, but installs to --user of default python

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

**===============================================================================**
                                **Running Demos**
**===============================================================================**

python load_sample_networks.py:
    #This will load networks into the work area (saved_networks)
    #If this will cause existing networks to be overwritten, confirmation will be required
    #This must be used before using run_test the first time, unless run_training_demo has been
    #run for the specified environment
    
python run_test_demo.py [--env <env_name>]:
    #By default, this will run the MsPacman demo, as a decent sample network has been provided
    #It will generate a .gif file of the episode (See final print statements)
    #Gifs can't be displayed on the server, so we recommend copying and running them on another machine
    
    #You can choose to run a different environment by adding a --env <environment_name> flag
    #You can choose how many episodes to run (only best will be rendered) by adding --n_iter <#> flag
    
    #Supported environments are: snake,MsPacman-v0,Asteroids-v0,CarRacing-v0
    
    #NOTE: We do not have a method of displaying graphics from the server. Results will be saved to a .gif file
    #Which can be loaded to another computer to play
    #Also NOTE: CarRacing-v0 in particular only works when rendering is enabled, so it will not run on the server
    
python run_training_demo.py [--env <env_name>]:
    #By default, this will run train MsPacman, as a decent sample network has been provided
    #network
    
    #You can choose to run a different environment by adding a --env <environment_name> flag
    #Supported environments are: snake,MsPacman-v0,Asteroids-v0,CarRacing-v0
    
    
python run_test_demo.ipynb:
    #Serves same purpose as run_test_demo.py, but as a notebook. We were unable to enable animations on the notebooks,
    #So the only difference here is that the final frame of the played game is displayed on the notebook
    
    #NOTE: This has not been thoroughly tested and the setup.sh scripts will not configure these notebooks properly,
    because the notebook session is loaded before the scripts can run, and we do not know how to open new notebook session.
    
    #Instead of command line parameters, cell 2 has 2 variables at the top: env and n_iter, these have the same usage as run_test_demo.py
   
python run_training_demo.ipynb:
    #Serves same purpose as run_training_demo.py, but as a notebook.
    There is no useful difference between this file and run_training_demo.py. We recommend using run_training_demo.py instead of this, as
    the print statements are clearer and more informative.
    
    #NOTE: This has not been thoroughly tested and the setup.sh scripts will not configure these notebooks properly,
    because the notebook session is loaded before the scripts can run, and we do not know how to open new notebook session.
    
    #Instead of command line parameters, cell 2 has 1 variable at the top: env, it has the same usage as run_training_demo.py args

**===============================================================================**
                                **Graphing score results**
**===============================================================================**
    
python generate_graphs.py --dir <dir_path>
    #This function will grab the .rewards and .qs files from the specified directory and plot the results
    #Plots will be saved as a .png if possible
    
    #If the --save_rewards flag is set during training, these files will be generated in appropriate saved_networks directory
    #An example would be python generate_graphs.py --dir ./saved_networks/MsPacman-v0_PacNet
    
**===============================================================================**
                                **Using the Tool**
**===============================================================================**
    #The main program file is deepQ.py, it accepts the following command line arguments:
    
    Required:
      --env <string>    #The environment you want to run, supports {snake,MsPacman-v0,CarRacing-v0,Asteroids-v0}
      --proto <string>  #Path to .prototxt file ex: cfn/MsPacman-v0.prototxt
      
    Recommended: (Do not use them all, but keep them in mind)
      --fresh           #Include to wipe the existing network (If there is one) and begin anew
      --save_rewards    #Include to save the .reward and .qs files used in plotting improvement over time
      --test            #Include to generate an demo run instead of a training run (Generates Example_run.gif)
      --max_neg_reward_steps <int> #Include to stop run early if too many consecutive negative rewards occur
      --game_skip <int>  #Number of frames to skip every time environment is reset
      
    Other:
      --n_steps <int>    #Number of training steps to take (Training will not occur if this number is less that # already completed)
      
      --n_prev_states <int>         #Number of previous states to hold in memory, network will perform poorly if this is too small to represent environment
      --checkpoint_interval <int>   #Interval at which a checkpoint of the network is saved
      --target_update_interval<int> #Interval at which agent is copied to target agent
      
      --learning_interval <int>     #Interval at which network should learn
      --minibatch_size <int>        #Number of samples target network should examine when estimating Q
      
      --momentum <float>            #Momentum value passed to momentum SGD optimizer
      --learning_rate <float>       #Learning rate passed to momentum SGD optimizer
      
      --epsilon_min <float>         #Minimum probability of taking random action during training
      --epsilon_max <float>         #Maximum probability of taking random action during training
      --epsion_steps <int>          #Number of steps to linearly go from epsilon max to epsilon min
      
      --discount <float>            #Amount to discount the target estimate Q

      
    Example uses:

      python deepQ.py --env MsPacman-v0 --proto cfn/MsPacman-v0.prototxt --learning_interval 4 --save_rewards
      
      python deepQ.py --env MsPacman-v0 --proto cfn/MsPacman-v0.prototxt --test
      
      python deepQ.py --env CarRacing-v0 --proto cfn/CarRacing-v0.prototxt --max_neg_reward_steps 150 --save_rewards --fresh

**===============================================================================**
                                **Miscellaneous References**
**===============================================================================**

Blog page about deep-Q learning: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/deep_q_learning.html

Longer paper about Deep-Q learning: http://proceedings.mlr.press/v48/gu16.pdf

Good source for convolutional Neural Nets: http://cs231n.github.io/convolutional-networks/#conv

AI_Gym (May allow for more complex games): https://gym.openai.com/

Arcade_Learning_environment (Specific to Atari games): https://github.com/mgbellemare/Arcade-Learning-Environment

Git repository for a3c using pytorch (not Deep-Q): https://github.com/ikostrikov/pytorch-a3c
