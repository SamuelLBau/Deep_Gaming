import argparse
import gym
import numpy as np

import matplotlib
matplotlib.use('Agg')#Fixes error on DSMP server
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from deepQ import deepQ


#These are used for image pre-processing
frame_list = []
num_frames = 0

def run_tests(args,gif_path,num_tests=10):
    network= deepQ(**args)

    max_anim = None
    max_score = -1e9
    for i in range(num_tests):
        print("Running test %d of %d"%(i+1,num_tests))
        [anim, score] = network.run_test()
        if score > max_score:
            max_anim = anim
            max_score = score
    network.play_test(max_anim,gif_path)

    del network


def mspacman_preprocess_func(frame):
    mspacman_c = 448 #210 + 164 + 74
    img = frame[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_c] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80,1)

def carracing_preprocess_func(frame):
    global frame_list
    global frame_num
    img =  frame[:,:,0] * 0.2125
    img += frame[:,:,1] * 0.7154
    img += frame[:,:,2] * 0.0721
    img -= 1
    if len(frame_list)==0:
        frame_list = np.array([img,img,img,img])
    else:
        frame_list = np.array([frame_list[1],frame_list[2],frame_list[3],img])
    img = frame_list[1] + frame_list[2]*2 + frame_list[3] * 3 + img * 4
    img = img/10
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    
    return img.reshape(96, 96,1)
def configure_test_1():
    args = {}
    args["game_type"] = "MsPacman-v0"
    args["env"] = gym.make(args["game_type"])
    args["proto"] = "cfn/MsPacman-v0_history.prototxt"
    args["action_space"] = list(range(9))
    #print(args["action_space"])
    args["preprocess_func"] = mspacman_preprocess_func
    #args["n_episodes"]=4000000
    #args["game_skip"] = 80
    #args["minibatch_size"] = 32
    #args["fresh"] = False
    #args["save_rewards"] = True
    #args["learning_interval"] = 4
    return args
    
def configure_test_2():
    args = {}
    args["game_type"] = "CarRacing-v0"
    args["env"] = gym.make(args["game_type"])
    args["proto"] = "cfn/CarRacing-v0.prototxt"
    args["preprocess_func"] = carracing_preprocess_func
    
    #Discretize the action space
    render = True
    range0 = [-1,0,1]
    range1 = [1,0]
    range2 = [.2,0]
    action_space = []
    for i in range0:
        for j in range1:
            for k in range2:
                action_space.append([i,j,k])
    args["action_space"] = action_space
    
    #args["n_episodes"]=4000000
    #args["game_skip"] = 50
    #args["minibatch_size"] = 50
    #args["fresh"] = False
    #args["save_rewards"] = True
    #args["learning_interval"] = 1
    return args

    
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run a convolutional neural net on an openAI gym environment.')
    parser.add_argument("--env",type=str,help="Select a prototxt file to load up",required=False,default="MsPacman-v0")
    args = parser.parse_args()
    env = args.env

    if env == "MsPacman-v0":
        print("Running MsPacman test")
        args = configure_test_1()
        run_tests(args,"Mspacman_Example.gif",num_tests=1)
    elif env == "CarRacing-v0":
        print("Running CarRacing test")
        try:
            args = configure_test_2()
            run_tests(args,"CarRacing_Example.gif",num_tests=1)
        except Exception as e:
            print("Failed to run CarRacing example,",str(e))
    else:
        print("Unsupported test environment %s"%(env))





