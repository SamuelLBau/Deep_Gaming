virtual_env="Deep_Gaming_GPU"
#conda create -n ${virtual_env} python=3.5.5
#source activate ${virtual_env}
pip install --user numpy
pip install --user matplotlib
pip install --user imageio
pip install --user tensorflow-gpu==1.4.0-rc1
pip install --user gym
pip install --user cmake
pip install --user gym[atari]
#conda install --user -c https://conda.anaconda.org/kne pybox2d
#echo "Please run <source activate" ${virtual_env} ">"
