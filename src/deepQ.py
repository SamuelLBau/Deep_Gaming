import numpy
from snake_game import snake_game
import tensorflow as tf
import os

import random
import numpy as np
from collections import deque

from caffe_builder import build_CNN

#Used for display
import sys
import time



#Assumes game_env has the following funtions:
#step(control) Accepts an integer 0->#controls-1, returns (score,done)
#reset(render=False,save=False) (resets the environment), sets the render variable(For final display)
#get_image() gets a numpy array of the game image
#get_image_dims() #Returns dimensions of game screen
#get_num_controls() #Returns number of controls

#CNN_struct

class deepQ():
    #[80,80],4,4,
    def __init__(self,game_env,caffe_file,cost_struct,num_actions,n_episodes=100,e_init=.5,e_final=.05,gamma=.99,
        n_obs_timesteps=500,n_explore_timesteps=500,n_prev_states=590000,minibatch_size=32,
        frames_per_action=1,game_name="name_unspecified",preprocess_func=None):
        self.env = game_env
        self.num_actions = num_actions
        self.n_episodes = n_episodes
        self.e_init = e_init
        self.e_final = e_final
        self.gamma = gamma
        self.n_obs_timesteps = n_obs_timesteps
        self.n_explore_timesteps = n_explore_timesteps
        self.n_prev_states = n_prev_states
        self.minibatch_size = minibatch_size
        self.frames_per_action = frames_per_action
        self.game_name = game_name
        self.preprocess_func = preprocess_func
        build_success = True
        try:
            self.game_name,self.network_input,self.network_output,self.network_readout = \
                build_CNN(caffe_file=caffe_file)
                #self.build_CNN_old(board_size,n_board_frames,num_actions,CNN_struct,cost_struct)
        except Exception as ex_val:
            error_message = str(ex_val)
            print(  "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n" + \
                    "Error during network generation: (%s)\n"%(error_message) + \
                    "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
            build_success = False
        if build_success:
            try:
                #TODO: Consider creating cost function builder
                # define the cost function
                with tf.variable_scope("train"):
                    self.actions = tf.placeholder("float", [None, num_actions])
                    self.outputs = tf.placeholder("float", [None])
                    readout_action = tf.reduce_sum(tf.multiply(self.network_readout, self.actions), reduction_indices = 1)
                    cost = tf.reduce_mean(tf.square(self.outputs - readout_action))
                    self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
                print("FINISHED BUILD")
            except Exception as ex_val:
                error_message = str(ex_val)
                print(  "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n" + \
                        "Error during cost function generation: (%s)\n"%(error_message) + \
                        "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
                build_success = False


        if build_success:
            print("CNN built succesfully")
            print("Input shape = %s"%(self.network_input.shape))
            print("Output shape = %s"%(self.network_output.shape))
            print("Readout shape = %s"%(self.network_readout.shape))

        else:
            raise Exception("Error during CNN generation, see above messages")
    def train(self):
        session = tf.InteractiveSession()

        self.train_CNN(self.env,session,self.network_input,
            self.network_output,self.network_readout)
        #self.asri_train_network(self.env,self.network_input,self.network_readout,self.network_output,session)


        #s, readout, h_fc1=self.asri_create_network()
        #self.asri_train_network(self.env,s,readout,h_fc1,session)
    def train_CNN(self,env,session,network_input,network_output,network_readout):
        env.reset()
        input_layer_shape = network_input.shape
        self.n_board_frames = input_layer_shape[3]
        single_board_frame_shape = [input_layer_shape[1],input_layer_shape[2],1]
        n_board_frames_less_1 = self.n_board_frames-1

        prev_frames = deque()
        cur_frame = env.render(mode="rgb_array")
        if not self.preprocess_func is None:
            cur_frame = self.preprocess_func(cur_frame)
        frame_stack = []
        for i in range(self.n_board_frames):
            frame_stack.append(cur_frame)
        cur_state = np.stack(frame_stack, axis = 2)

        if not os.path.isdir("saved_networks"):
            os.mkdir("saved_networks")
        #TODO: RE-WRITE everything below this point
        checkpoint_path = str("saved_networks/%s"%(self.game_name))
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            #from tensorflow.python.tools import inspect_checkpoint as chkp
            #print("HERE")
            #chkp.print_tensors_in_checkpoint_file(checkpoint_path+str("\%s-1"%(self.game_name)), tensor_name='', all_tensors=True)
            #print("HERE2")
            #return
            checkpoint_saver.restore(session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

        else:
            print("Could not find old network weights")
        e = self.e_init
        t = 0
        num_finished_eps=0
        start_time = time.time()
        while True: #Continue learning until program quits
            # choose an action e greedily
            readout_t = self.network_readout.eval(feed_dict = {self.network_input : [cur_state]})[0]
            cur_action = np.zeros([self.num_actions])
            action_index = 0
            if random.random() <= e or t <= self.n_obs_timesteps:
                action_index = random.randrange(self.num_actions)
                cur_action[action_index] = 1
            else:
                #print("READOUT",readout_t)
                #print("SHAPE",readout_t.shape)
                action_index = np.argmax(readout_t)
                if(abs(np.max(readout_t)- np.min(readout_t)) < 1):
                    action_index = random.randrange(self.num_actions)
                #print("INDEXVAL",action_index)
                cur_action[action_index] = 1

            # scale down epsilon
            if e > self.e_final and t > self.n_obs_timesteps:
                e -= (self.e_init - self.e_final) / self.n_explore_timesteps

            for i in range(0, self.frames_per_action):
                # run the selected action and observe next state and reward
                next_frame,cur_reward, terminal, info = env.step(action_index)
                next_frame = self.preprocess_func(next_frame)
                next_frame = np.reshape(next_frame, single_board_frame_shape)
                if self.n_board_frames > 1:
                    next_state = np.append(next_frame, cur_state[:,:,0:n_board_frames_less_1], axis = 2)
                else:
                    next_state = next_frame
                # store the transition in prev_frames
                prev_frames.append((cur_state, cur_action, cur_reward, next_state, terminal))
                if len(prev_frames) > self.n_prev_states:
                    prev_frames.popleft()
            # only train if done observing
            if t > self.n_obs_timesteps:
                # sample a minibatch to train on
                minibatch = random.sample(prev_frames, self.minibatch_size)
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                output_batch = []
                readout_j1_batch = network_readout.eval(feed_dict = {network_input : s_j1_batch})
                for i in range(0, len(minibatch)):
                    # if terminal only equals reward
                    #print(minibatch)
                    if minibatch[i][4]:
                        output_batch.append(reward_batch[i])
                    else:
                        output_batch.append(reward_batch[i] + self.gamma * np.max(readout_j1_batch[i]))

                # perform gradient step
                self.train_step.run(feed_dict = {
                    self.outputs : output_batch,
                    self.actions: action_batch,
                    network_input : s_j_batch})
            if terminal:
                num_finished_eps+=1
                print(num_finished_eps,cur_reward)
                env.render(mode="rgb_array")
                env.reset()
            

            # update the old values
            cur_state = next_state
            t += 1

            # save progress every 10000 iterations
            if t % 2000 == 0:
                checkpoint_saver.save(session, checkpoint_path + "/" + self.game_name, global_step = 1)
            # print info
            state = ""
            if t <= self.n_obs_timesteps:
                state = "observe"
            elif t > self.n_obs_timesteps and t <= self.n_obs_timesteps + self.n_explore_timesteps:
                state = "explore"
            else:
                state = "train"
            if t % 500 == 0:
                end_time=time.time()
                steps_per_second = 500.0 / (end_time-start_time)
                print("TIMESTEP", t, "/ Steps/s", steps_per_second, "/ EPSILON", e, "/ ACTION", action_index, "/ REWARD", cur_reward, "/ Q_MAX %e" % np.max(readout_t))
                start_time = time.time()
        pass

if __name__ == "__main__":
    env = None
    import sys
    preprocess_func = None
    if len(sys.argv) < 2:
        raise Exception("Path of caffe file must be passed in")
    elif len(sys.argv) == 2:
        #If 1 arg passed in, is assumed to be snake
        proto_file = sys.argv[1]
        env = snake_game(board_size=[25,25],render=False) #This must match the shape in the supplied prototxt file
        num_actions = 4
        if not os.path.isfile(proto_file):
            raise Exception("Caffe file not found: %s"%(str(e)))
    else:
        #If more than 1 arg passed in, assumed to be python deepQ.py <prototxt file> <gamename>
        proto_file = sys.argv[1]
        game_type  = sys.argv[2]
        if game_type == "snake":
            env = snake_game(board_size=[25,25],render=False) #This must match the shape in the supplied prototxt file
            num_actions = 4
        else:
            import gym
            if game_type == "MsPacman-v0":
                def preprocess_func(frame):
                    mspacman_c = 448 #210 + 164 + 74
                    img = frame[1:176:2, ::2] # crop and downsize
                    img = img.sum(axis=2) # to greyscale
                    img[img==mspacman_c] = 0 # Improve contrast
                    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
                    return img.reshape(88, 80)
            
            
            env = gym.make(game_type)
            num_actions = 9     
    cost_struct = []

    network = deepQ(env,proto_file,cost_struct,num_actions,game_name=game_type,preprocess_func=preprocess_func)
    network.train()