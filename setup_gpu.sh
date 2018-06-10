virtual_env="Deep_Gaming_GPU"
conda create -n ${virtual_env} python=3.5.5
source activate ${virtual_env}
pip install numpy
pip install matplotlib
pip install tensorflow-gpu
pip install gym
pip install cmake
pip install gym[atari]
conda install -c https://conda.anaconda.org/kne pybox2d
echo "Please run <source activate ${virtal_env}>"
