virtual_env="Deep_Gaming_GPU"
conda create -n ${virtual_env} python=3.5.5
source activate ${virtual_env}
pip install numpy
pip install matplotlib
pip install imageio
pip install tensorflow-gpu==1.4.0-rc1
pip install gym
pip install cmake
pip install gym[atari]
conda install -c https://conda.anaconda.org/kne pybox2d
source deactivate
ipython kernel install --user --name=${virtual_env}
echo "Please disconnect, then reconnect to server to restart Jupyter session"
echo "Please run <source activate" ${virtual_env} ">"
