import argparse
import os
import struct
import numpy as np

import matplotlib.pyplot as plt

def read_ints(file_path):
	int_arr = []
	with open(file_path,"rb") as file:
		data = file.read()
	
	ind = 0
	num_bytes = len(data)
	while ind+4 <= num_bytes:
		val = struct.unpack("i",data[ind:ind+4])[0]
		int_arr.append(val)
		ind += 4
	return int_arr

def read_doubles(file_path):
	double_arr = []
	with open(file_path,"rb") as file:
		data = file.read()
	
	ind = 0
	num_bytes = len(data)
	while ind+8 <= num_bytes:
		val = struct.unpack("d",data[ind:ind+8])[0]
		double_arr.append(val)
		ind += 8
	return double_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a convolutional neural net on an openAI gym environment.')
    parser.add_argument("--dir",type=str,help="Select a directory to pull data from",required=True)

    args = parser.parse_args()

    path_prefix = args.dir + "/" + os.path.basename(args.dir)

    reward_file = path_prefix + ".rewards"
    q_file = path_prefix + ".qs"

    
    rewards = np.array(read_ints(reward_file))
    qs = np.array(read_doubles(q_file))
    
    rewards_average = np.convolve(rewards,np.ones((10))/10,mode="same")
    qs_average		= np.convolve(qs,np.ones((10))/10,mode="same")

    ax0,=plt.plot(rewards[1::50])
    ax1,=plt.plot(rewards_average[1::50])
    plt.legend([ax0,ax1],["Score","Moving average 10"])
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.title("MsPacman Score")
    plt.grid()
    plt.figure()

    ax0,=plt.plot(qs[1::50])
    ax1,=plt.plot(qs_average[1::50])
    plt.legend([ax0,ax1],["Q sum","Moving average 10"])
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.title("MsPacman Total Qs")
    plt.grid()
    plt.show()