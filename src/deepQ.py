import numpy
from snake_game import snake_game
import tensorflow as tf
import os
import gym

import sys
import argparse

import random
import numpy as np
from collections import deque

from caffe_builder import build_CNN

import struct

#Used for display
import time
import datetime

import matplotlib
matplotlib.use('Agg')#Fixes error on DSMP server
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import imageio #Backup gif writer if matplotlib fails

from copy import deepcopy
##DEBUGGING
from math import isnan



#Assumes env has the following funtions:
#step(control) Accepts an integer 0->#controls-1, returns (score,done)
#reset(render=False,save=False) (resets the environment), sets the render variable(For final display)
#get_image() gets a numpy array of the game image
#get_image_dims() #Returns dimensions of game screen
#get_num_controls() #Returns number of controls

#CNN_struct

#NOTE:

#NOTE: This loss function and training algorithm are heavily modified versions of that found at
#https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb

class deepQ():
    #[80,80],4,4,
    def __init__(self,game_type,env,proto,action_space,preprocess_func,
        game_skip=0,n_steps=10000000,
        momentum=.95,learning_rate=.001,discount=.99,
        epsilon_min=.1,epsilon_max=1.0,epsilon_steps=2000000,
        n_prev_states=100000,checkpoint_interval=500,target_update_interval=5000,
        learning_interval=1,minibatch_size=50,
        save_rewards=False,fresh=False,render=False,max_neg_reward_steps=1e9):
        
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.game_type = game_type
        self.env = env
        self.proto = proto
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.preprocess_func = preprocess_func

        self.game_skip = game_skip
        self.n_steps = n_steps

        self.momentum = momentum                #does not need to be saved
        self.learning_rate=learning_rate        #does not need to be saved
        self.discount = discount

        self.n_prev_states=n_prev_states
        self.checkpoint_interval = checkpoint_interval
        self.target_update_interval = target_update_interval
        
        self.learning_interval = learning_interval
        self.minibatch_size = minibatch_size

        self.epsilon_min=epsilon_min
        self.epsilon_max=epsilon_max
        self.epsilon_steps=epsilon_steps

        self.save_rewards = save_rewards
        self.fresh=fresh
        self.render=render
        self.max_neg_reward_steps = max_neg_reward_steps

        build_success = True
        try:
            
            self.network_name,self.online_input,self.online_output,self.online_readout,self.online_vars = \
                build_CNN("q_networks/online",caffe_file=proto)
            self.network_name,self.target_input,self.target_output,self.target_readout,self.target_vars = \
                build_CNN("q_networks/target",caffe_file=proto,input = self.online_input)
            #self.network_name,self.truth_input,self.truth_output,self.truth_readout,self.truth_vars = \
            #    build_CNN("q_networks/truth",caffe_file=proto,input = self.online_input)
           
            self.copy_ops = [target_var.assign(self.online_vars[var_name])
                for var_name, target_var in self.target_vars.items()]
            
            self.copy_online_to_target = tf.group(*self.copy_ops)
            for val in self.online_vars:
                print(val,self.online_vars[val])
            print("DONE")
        except Exception as ex_val:
            error_message = str(ex_val)
            print(  "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n" + \
                    "Error during network generation: (%s)\n"%(error_message) + \
                    "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
        
        try:
            #TODO: Consider creating loss function builder
            self.n_outputs = self.target_output.shape[1]

            #Select optimizer type
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)

            #NOTE: This is the same loss function found at, though the implementation has been changed:
            #https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
            with tf.variable_scope("train") as scope:
                self.action = tf.placeholder(tf.int32, shape=[None])
                self.sampled_vals = tf.placeholder(tf.float32, shape=[None, 1])
                selected_qs = self.online_output * tf.one_hot(self.action, self.n_outputs)

                try:
                    q_values = tf.reduce_sum(selected_qs,keepdims=True,axis=1)
                except Exception as e:
                    #Older versions of tensorflow use keep_dims
                    q_values = tf.reduce_sum(selected_qs,keep_dims=True,axis=1)
                error = tf.abs(self.sampled_vals - q_values)

                less_1_error = tf.square(error)
                more_1_error = 2.0 * error - 1.0

                #If error is < 1, square the error
                #If error > 1 double the error - 1
                less_vals = tf.less(error,1.0)
                correct_err = tf.where(less_vals,less_1_error,more_1_error)
                self.loss = tf.reduce_mean(correct_err)

                self.training_step = tf.Variable(0, trainable=False, name='global_step')
                self.training_op = optimizer.minimize(self.loss, global_step=self.training_step)

            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        except Exception as ex_val:
            error_message = str(ex_val)
            print(  "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n" + \
                    "Error during loss function generation: (%s)\n"%(error_message) + \
                    "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")

        self.checkpoint_path = str("./saved_networks/%s_%s/%s_%s.ckpt"%(self.game_type,self.network_name,self.game_type,self.network_name))
        print("CNN built succesfully")
        print("Input shape = %s"%(self.online_input.shape))
        print("Output shape = %s"%(self.online_output.shape))
        #print("Readout shape = %s"%(self.online_readout.shape))

    def write_settings(self,file_path):
        with open(file_path,"w") as file:
            file.write("python deepQ.py ")
            file.write(str("--env %s "%(self.game_type)))
            file.write(str("--proto %s "%(self.proto)))

            file.write(str("--game_skip %d "%(self.game_skip)))
            file.write(str("--n_steps %d "%(self.n_steps)))

            file.write(str("--momentum %f "%(self.momentum)))
            file.write(str("--learning_rate %f "%(self.learning_rate)))
            file.write(str("--discount %f "%(self.discount)))

            print(self.epsilon_min,type(self.epsilon_min))
            file.write(str("--epsilon_min %f "%(self.epsilon_min)))
            file.write(str("--epsilon_max %f "%(self.epsilon_max)))
            file.write(str("--epsilon_steps %d "%(self.epsilon_steps)))

            file.write(str("--n_prev_states %d "%(self.n_prev_states)))

            file.write(str("--checkpoint_interval %d "%(self.checkpoint_interval)))
            file.write(str("--target_update_interval %d "%(self.target_update_interval)))
            file.write(str("--learning_interval %d "%(self.learning_interval)))

            file.write(str("--minibatch_size %d "%(self.minibatch_size)))
            file.write(str("--max_neg_reward_steps %d "%(self.max_neg_reward_steps)))
            if self.save_rewards:
                file.write("--save_rewards ")




    def select_action(self,q_values,step_num):

        #y  =                 m*x + b
        e_p = -(self.epsilon_max-self.epsilon_min) * (float(step_num)/self.epsilon_steps) + self.epsilon_max
        e = max(self.epsilon_min,e_p)
        if np.random.rand() < e:
            return np.random.randint(self.n_outputs),e # random action
        else:
            return np.argmax(q_values),e # optimal action
    
    
    def reset_env(self):
        obs = self.env.reset()
        for skip in range(self.game_skip): # skip the start of each game
            obs, reward, done, info = self.env.step(self.action_space[0])
            if self.render:
                self.env.render()
        return self.preprocess_func(obs)
    def train(self):

        #NOTE: This is a very heavily modifed version of the framework found at:
        #https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
        self.env.reset()
       
        run_string = "%s_%s"%(self.game_type,self.network_name)
        #writer = tf.summary.FileWriter("./tf_log.log", graph=self.session.graph)
        if not os.path.isdir("./saved_networks"):
            os.mkdir("./saved_networks")
        if not os.path.isdir(str("./saved_networks/%s"%(run_string))):
            os.mkdir(str("./saved_networks/%s"%(run_string)))
        reward_data_path = str("./saved_networks/%s/%s.rewards"%(run_string,run_string))
        max_q_path = str("./saved_networks/%s/%s.qs"%(run_string,run_string))

        cmd_file_path = str("./saved_networks/%s/%s.cmd"%(run_string,run_string))
        self.write_settings(cmd_file_path)


        prev_frames = deque(maxlen=self.n_prev_states)

        self.session = tf.InteractiveSession()
        if os.path.isfile(self.checkpoint_path + ".index") and not self.fresh:
            self.saver.restore(self.session, self.checkpoint_path)
            #self.copy_online_to_target.run()
        else:
            self.initializer.run()
            self.copy_online_to_target.run()
        if self.save_rewards:
            if (self.training_step.eval() == 0 and os.path.isfile(reward_data_path)) or self.fresh:
                reward_file = open(reward_data_path,"wb")
            else:
                reward_file = open(reward_data_path,"ab")
            if (self.training_step.eval() == 0 and os.path.isfile(max_q_path)) or self.fresh:
                q_file = open(max_q_path,"wb")
            else:
                q_file = open(max_q_path,"ab")
        #start_time = time.time()
        

        loss = float("inf")
        done = False
        game_length = 0
        total_reward = 0
        total_max_q = 0
        mean_max_q = 0.0
        avg_reward_list = []
        avg_reward = 0
        neg_reward_step_count = 0
        cur_iter = 0  # game iterations
        time_str = "inf"
        self.reset_env()

        #This pre-loads frames into the memory
        for i in range(self.minibatch_size+1):
            obs, reward, done, info = self.env.step(self.action_space[0])
            next_state = self.preprocess_func(obs)
            if i == 0:
                state = next_state
            prev_frames.append((state, 0, reward, next_state, 1.0 - done))
            state = next_state
        action=0
        start_time = time.time()

        if not self.training_step.eval() < self.n_steps+1:
            print("Warning, no iterations will be run. n_steps= %d, cur_step = %d"%(self.n_steps,self.training_step.eval()-1))
        
        while self.training_step.eval() < self.n_steps+1:
            cur_step = self.training_step.eval()
            cur_iter += 1

            #if iteration%100 == 0:
            #    plt.imshow(obs)
            #    print(done)
            #    plt.show()
            # Online DQN evaluates what to do
            #print("\nINPUT SHAPE",self.online_input.shape)
            #print("PROCESS_SHAPE",state.shape)
            #self.copy_online_to_truth.run()
            q_values = self.online_output.eval(feed_dict={self.online_input: np.array([state])})
            #temp_q_values = self.truth_output.eval(feed_dict={self.truth_input: np.array([state])})
            #print("q_VALUES",q_values)
            action,e = self.select_action(q_values, cur_step)

            # Online DQN plays
            obs, reward, done, info = self.env.step(self.action_space[action])
            if self.render:
                self.env.render()
            next_state = self.preprocess_func(obs)

            # Let's memorize what happened
            prev_frames.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_reward += reward
            total_max_q += q_values.max()
            game_length += 1

            if reward <=0.0:
                neg_reward_step_count += 1
                if neg_reward_step_count >= self.max_neg_reward_steps:
                    done = True
                    neg_reward_step_count = 0
            else:
                neg_reward_step_count = 0


            #Update score / q files, reset environment
            if done:
                if self.save_rewards:
                    reward_file.write(int(total_reward).to_bytes(4, byteorder='little', signed=True))
                    reward_file.flush()
                    
                    ba = bytearray(struct.pack("d", total_max_q)) 
                    q_file.write(ba)
                    q_file.flush()

                if len(avg_reward_list) >= 10:
                    del avg_reward_list[0]
                avg_reward_list.append(total_reward)
                avg_reward = np.mean(avg_reward_list)
                
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0
                total_reward = 0
                neg_reward_step_count = 0
                self.reset_env()


            if cur_iter % self.learning_interval != 0:
                continue #Optionally skip certain frames
            
            samples = random.sample(prev_frames, min(len(prev_frames),self.minibatch_size))
            
            sampled_state_vals =       np.array([samp[0] for samp in samples])
            sampled_action_vals =      np.array([samp[1] for samp in samples])
            sampled_rewards =          np.array([samp[2] for samp in samples])
            sampled_next_state_vals =  np.array([samp[3] for samp in samples])
            sampled_continues =        np.array([samp[4] for samp in samples])

            next_qs = self.target_output.eval(
                feed_dict={self.target_input: sampled_next_state_vals})
            max_next_qs = np.max(next_qs, axis=1)

            sampled_vals =np.expand_dims(sampled_rewards + sampled_continues * \
                self.discount * max_next_qs,1)

            GARBAGE, loss = self.session.run([self.training_op, self.loss], feed_dict={
                self.online_input: sampled_state_vals, self.action: sampled_action_vals, self.sampled_vals: sampled_vals})

            # Regularly copy the online DQN to the target DQN
            if cur_step % self.target_update_interval == 0:
                self.copy_online_to_target.run()

            # And save regularly
            if cur_step % self.checkpoint_interval == 0:
                self.saver.save(self.session, self.checkpoint_path)
                #print(q_values)
                #print(np.argmax(q_values))

            if cur_step % 100 == 0:
                dif_time = time.time() - start_time
                start_time = time.time()
                est_time = dif_time * (self.n_steps - cur_step) / 100.0 #Calculated every 100 steps
                time_str = str(datetime.timedelta(seconds=est_time)).split(".")[0]

            print_str = "                                                                                       \r"
            print_str += "%s: Step %8d of %8d (%2.4f%%),\tCur_Reward %.3f,\tAverage Reward %.3f,\tEst. t remain: %s\t,e %2.2f"%(run_string,cur_step,self.n_steps,100.0*cur_step/self.n_steps,total_reward,avg_reward,time_str,e)
            sys.stdout.write(print_str)
            sys.stdout.flush()

    
    def run_test(self,max_steps=20000):
        print("Running test")
        frames = []
        tot_reward=0
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)

            obs = self.env.reset()
            neg_reward_step_count=0
            for step in range(max_steps):
                state = self.preprocess_func(obs)
                # Online DQN evaluates what to do
                q_values = self.online_output.eval(feed_dict={self.online_input: [state]})
                #print(q_values)
                action = np.argmax(q_values)
                #print(q_values)
                #print(self.action_space[action])
                #print(action)
                #print(step)
                y_val = [np.max(q_values)]

                # Online DQN plays
                obs, reward, done, info = self.env.step(self.action_space[action])

                tot_reward += reward
                img = self.env.render(mode="rgb_array")
                frames.append(img)

                if reward <=0.0:
                    neg_reward_step_count += 1
                    if neg_reward_step_count >= self.max_neg_reward_steps:
                        done = True
                        neg_reward_step_count = 0
                else:
                    neg_reward_step_count = 0

                if done:
                    break

            else:
                print("NOTE: game did not end with done flag")


        print("Rendering test")

        def update_scene(num, frames, patch):
            patch.set_data(frames[num])
            return patch,
        def plot_animation(frames, repeat=False, interval=40):
            plt.close()  # or else nbagg sometimes plots in the previous cell
            fig = plt.figure()
            patch = plt.imshow(frames[-1])
            plt.axis('off')
            return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
        return [frames,reward,frames]

    def play_test(self,anim,frames,gif_path=None):
        try:
            plt.imshow(anim)
            plt.show()
            if not gif_path is None:
                anim.save(gif_path)
        except Exception as e:
            print("Failed to display animation %s"%(str(e)))
            if not gif_path is None:
                print("Could not save gif, attempting to use imageio")
                try:
                    with imageio.get_writer(gif_path,mode="I",fps=25) as gif_writer:
                        for frame in frames:
                            gif_writer.append_data(frame)
                except Exception as e2:
                    print("ImageIO Writer also failed"%(str(e2)))

        
        plt.show()

frame_list = []
frame_num = 1
_preprocess_func = None
if __name__ == "__main__":
    global _preprocess_func
    env = None
    render=False

    parser = argparse.ArgumentParser(description='Run a convolutional neural net on an openAI gym environment.')
    parser.add_argument("--proto",type=str,help="Select a prototxt file to load up",required=True)
    parser.add_argument("--env",type=str,help="Select a openai environment to load",required=True)
    parser.add_argument("--game_skip",help="Number of frames to skip after env is reset",required=False,type=int,default=80)
    parser.add_argument("--n_steps",help="Number of steps to train",required=False,type=int,default=4000000)

    parser.add_argument("--momentum",help="Training momentum",required=False,type=float,default=.95)
    parser.add_argument("--learning_rate",help="Training rate",required=False,type=float,default=.001)
    parser.add_argument("--discount",help="Decay rate of older samples",required=False,type=float,default=.99)

    parser.add_argument("--epsilon_min",help="minimum random exploration probability",required=False,type=float,default=.1)
    parser.add_argument("--epsilon_max",help="maximum random exploration probability",required=False,type=float,default=1.0)
    parser.add_argument("--epsilon_steps",help="Steps to go from e_min to e_max",required=False,type=int,default=2000000)

    parser.add_argument("--n_prev_states",help="Number of previous states to store in memory",required=False,type=int,default=100000)
    parser.add_argument("--checkpoint_interval",help="How many steps before updating checkpoint",required=False,type=int,default=1000)
    parser.add_argument("--target_update_interval",help="How many steps before updating target",required=False,type=int,default=5000)


    parser.add_argument("--learning_interval",help="How many steps between learning step",required=False,type=int,default=4)
    parser.add_argument("--minibatch_size",help="How many states to learn off at the same time",required=False,type=int,default=32)

    parser.add_argument("--save_rewards",help="Run test, do not perform learning",required=False,action="store_true")#run_test = False
    parser.add_argument("--test",help="Run test, do not perform learning",required=False,action="store_true")#run_test = False
    parser.add_argument("--fresh",help="Ignore existing checkpoint files, restart learning",required=False,action="store_true")#run_test = False
    parser.add_argument("--max_neg_reward_steps",help="Reset if negative reward hit this many times",required=False,type=int,default=1e9)


    args = parser.parse_args()
    proto_file = args.proto
    game_type = args.env
    run_test = args.test

    print(args)
    if not os.path.isfile(proto_file):
        raise Exception("Caffe file not found: %s"%(proto_file))

    if game_type == "snake":
        env = snake_game(board_size=[25,25])
        action_space = list(range(4))
    else:
        env = gym.make(game_type)
        if hasattr(env.action_space,"n"):
            action_space = list(range(env.action_space.n))
        else:
            if game_type == "CarRacing-v0":
                env = env.env #gym tries to automatically reset after 1000 steps, I don't like this
                render = True
                range0 = [-1,0,1]
                range1 = [1,0]
                range2 = [.2,0]
                action_space = []
                for i in range0:
                    for j in range1:
                        for k in range2:
                            action_space.append([i,j,k])
            else:
                raise Exception("Issue with environment, could not detect number of actions")
            print("NUM ACT",action_space)
    

    if game_type == "MsPacman-v0":
        def temp_func(frame):
            mspacman_c = 448 #210 + 164 + 74
            img = frame[1:176:2, ::2] # crop and downsize
            img = img.sum(axis=2) # to greyscale
            img[img==mspacman_c] = 0 # Improve contrast
            img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
            return img.reshape(88, 80,1)
        _preprocess_func = temp_func
    elif game_type == "Asteroids-v0":
        def temp_func(frame):
            img = frame[34:210:2, ::2] # crop and downsize
            img = img.sum(axis=2) # to greyscale

            #plt.imshow(frame)
            #plt.figure()
            #plt.imshow(img)
            #plt.show()
            return img.reshape(88, 80,1)
        _preprocess_func = temp_func
    elif game_type == "CarRacing-v0":
        def temp_func(frame):
            img =  frame[:,:,0] * 0.2125
            img += frame[:,:,1] * 0.7154
            img += frame[:,:,2] * 0.0721
            img -= 1
            return img.reshape(96, 96,1)
        _preprocess_func = temp_func

    else:
        def temp_func(frame):
            return np.expand_dims(frame,2)
        _preprocess_func = temp_func
        print("WARNING: NO PREPROCESS_FUNC SPECIFIED")

    if "_history" in proto_file:
        def history_func(frame):
            global frame_list
            global frame_num
            global _preprocess_func
            img = _preprocess_func(frame)

            if len(frame_list)==0:
                frame_list = np.array([img,img,img,img])
            else:
                frame_list = np.array([frame_list[1],frame_list[2],frame_list[3],img])
            img = frame_list[1] + frame_list[2]*2 + frame_list[3] * 3 + img * 4
            img = img/10
            img = img.astype(np.int8)

            return img
        preprocess_func = history_func
    else:
        preprocess_func = _preprocess_func
    #raise Exception("TEMP")
    network = deepQ(game_type,env,proto_file,action_space,preprocess_func,
        game_skip=args.game_skip,n_steps=args.n_steps,
        momentum=args.momentum,learning_rate=args.learning_rate,discount=args.discount,
        epsilon_min=args.epsilon_min,epsilon_max=args.epsilon_max,epsilon_steps=args.epsilon_steps,
        n_prev_states=args.n_prev_states,checkpoint_interval=args.checkpoint_interval,target_update_interval=args.target_update_interval,
        learning_interval=args.learning_interval,minibatch_size=args.minibatch_size,
        save_rewards=args.save_rewards,fresh=args.fresh,render=render,max_neg_reward_steps=args.max_neg_reward_steps)

    if run_test:
        anim,score,frames = network.run_test()
        network.play_test(anim,frames,gif_path="Example_run.gif")
    else:
        network.train()