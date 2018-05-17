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
        frames_per_action=1,game_name="name_unspecified"):
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
        
        try:
            #TODO: Consider creating cost function builder
            # define the cost function
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
        cur_frame = env.get_image()
        frame_stack = []
        for i in range(self.n_board_frames):
            frame_stack.append(cur_frame)
        cur_state = np.stack(frame_stack, axis = 2)


        #TODO: RE-WRITE everything below this point
        checkpoint_path = str("saved_networks/%s"%(self.game_name))
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_saver.restore(session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
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
                cur_reward, terminal = env.step(action_index)
                next_frame = env.get_image()
                next_frame = np.reshape(next_frame, single_board_frame_shape)
                next_state = np.append(next_frame, cur_state[:,:,0:n_board_frames_less_1], axis = 2)

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
                    if minibatch[i][self.n_board_frames]:
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
                if num_finished_eps%1000==0:
                    env.reset(render=False)
                else:
                    env.reset(render=False)
            

            # update the old values
            cur_state = next_state
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                checkpoint_saver.save(session, checkpoint_path + "/" + self.game_name + '-dqn', global_step = 1)

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
                print "TIMESTEP", t, "/ Steps/s", steps_per_second, "/ EPSILON", e, "/ ACTION", action_index, "/ REWARD", cur_reward, "/ Q_MAX %e" % np.max(readout_t)
                start_time = time.time()
        pass
#Close deepQ
def call_asri_network_to_CNN_struct():
    return asri_network_to_CNN_struct()

if __name__ == "__main__":
    env = snake_game(board_size=[25,25],render=False) #This must match the shape in the supplied prototxt file
    num_actions = 4
    cost_struct = []

    import sys
    if len(sys.argv) < 2:
        raise Exception("Path of caffe file must be passed in")
    else:
        proto_file = sys.argv[1]
        if not os.path.isfile(proto_file):
            raise Exception("Caffe file not found: %s"%(str(e)))
    
    network = deepQ(env,proto_file,cost_struct,num_actions,game_name="tetris")
    network.train()



#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

def asri_network_to_CNN_struct():
    n_actions = 4
    '''
        This builds the following CNN architecture

    '''

    #W_conv1 = weight_variable([8, 8, 4, 32],.01)
    #b_conv1 = bias_variable(.01,[32])
    #h_conv1 = tf.nn.relu(conv2d(s, W_conv1, [1, 4, 4, 1]) + b_conv1)
    layer_1 = {}
    layer_1["type"]     = "relu"
    layer_1["B_val"]    = .01
    layer_1["B_shape"]  = [32]

    layer_1["function"]             = {}
    layer_1["function"]["name"]     = "conv2d"
    layer_1["function"]["stride"]   = [1, 4, 4, 1]
    layer_1["function"]["W_shape"]  = [8, 8, 4, 32]  
    layer_1["function"]["W_std"]    = .01

    #h_pool1 = max_pool_2x2(h_conv1,[1, 2, 2, 1],[1, 2, 2, 1])
    layer_2 = {}
    layer_2["type"]     = "pool"
    layer_2["strides"]  = [1, 2, 2, 1]
    layer_2["ksize"]    = [1, 2, 2, 1]

    #W_conv2 = weight_variable([4, 4, 32, 64],.01)
    #b_conv2 = bias_variable(.01,[64])
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1, 2, 2, 1]) + b_conv2)
    layer_3 = {}
    layer_3["type"] = "relu"
    layer_3["B_val"]    = .01
    layer_3["B_shape"]  = [64]

    layer_3["function"] = {}
    layer_3["function"]["name"] = "conv2d"
    layer_3["function"]["stride"]           = [1, 2, 2, 1]
    layer_3["function"]["W_shape"]          = [4, 4, 32, 64] 
    layer_3["function"]["W_std"]            = .01

    #W_conv3 = weight_variable([3, 3, 64, 64],.01)
    #b_conv3 = bias_variable(.01,[64])
    #h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, [1, 1, 1, 1]) + b_conv3)
    layer_4 = {}
    layer_4["type"] = "relu"
    layer_4["B_val"]    = .01
    layer_4["B_shape"]  = [64]

    layer_4["function"] = {}
    layer_4["function"]["name"] = "conv2d"
    layer_4["function"]["stride"]           = [1, 1, 1, 1]
    layer_4["function"]["W_shape"]          = [3, 3, 64, 64]
    layer_4["function"]["W_std"]            = .01

    #h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    layer_5 = {}
    layer_5["type"] = "reshape"
    layer_5["shape"]= [-1, 1600]

    #W_fc1 = weight_variable([1600, 512],.01)
    #b_fc1 = bias_variable(.01,[512])
    #h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    layer_6 = {}
    layer_6["type"] = "relu"
    layer_6["B_val"]    = .01
    layer_6["B_shape"]  = [512]

    layer_6["function"] = {}
    layer_6["function"]["name"] = "matmul"
    layer_6["function"]["W_shape"]          = [1600, 512]
    layer_6["function"]["W_std"]            = .01

    #W_fc2 = weight_variable([512, ACTIONS],.01)
    #b_fc2 = bias_variable(.01,[ACTIONS])
    #readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    layer_7 = {}
    layer_7["type"] = "matmul"
    layer_7["B_val"]    = .01
    layer_7["B_shape"]  = [n_actions]
    layer_7["W_shape"]  = [512, n_actions]
    layer_7["W_std"]    = .01

    return [layer_1,layer_2,layer_3,layer_4,layer_5,layer_6,layer_7]

'''
def build_CNN_old(self,board_shape,board_frames,num_actions,CNN_struct,cost_struct):
    self.cost_struct = cost_struct
    
        Builds the CNN based on a designated structure
        #TODO: write-out structure and valid inputs
        
        #TODO: Determine method of generating CNN_struct in above sections


    
    #------------BEGIN CREATING NETWORK
    network_input = tf.placeholder("float", [None, board_shape[0], board_shape[1], board_frames])

    current_parent = network_input
    for hidden_layer in CNN_struct[:-1]:
        current_parent = self.assemble_layer(current_parent,hidden_layer)

    network_output = current_parent
    readout_layer = self.assemble_layer(network_output,CNN_struct[-1])
    #TODO: Verify the number of output nodes == num_actions, throw error if otherwise

    #TODO: Consider creating cost function builder
    # define the cost function
    self.actions = tf.placeholder("float", [None, num_actions])
    self.outputs = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout_layer, self.actions), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(self.outputs - readout_action))
    self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    return network_input,network_output,readout_layer

def assemble_layer(self,current_parent,layer_info):
    cur_layer = None
    
    #-----------------Begin helper funtions-----------------------------------------
    def create_W(shape,std):
        return tf.Variable(tf.truncated_normal(shape, stddev = std))

    def create_B(val,shape):
        return tf.Variable(tf.constant(val, shape = shape))
    
    def create_matmul(parent,W_mat):
        return tf.matmul(parent, W_mat)
    def create_conv2d(parent, W_mat, strides):
        return tf.nn.conv2d(parent, W_mat, strides = strides, padding = "SAME")

    def create_pool2(parent,strides,ksize):
        return tf.nn.max_pool(parent, strides = strides,ksize = ksize, padding = "SAME")
    
    if "function" in layer_info:
        layer_function = layer_info["function"]
        function_name = layer_function["name"]
        if function_name == "conv2d":
            conv_stride = layer_function["stride"]
            W_shape = layer_function["W_shape"]
            W_std = layer_function["W_std"]
            W_var = create_W(W_shape,W_std)

            func = create_conv2d(current_parent,W_var,conv_stride)
        elif function_name == "matmul":
            W_shape = layer_function["W_shape"]
            W_std   = layer_function["W_std"]
            W_var = create_W(W_shape,W_std)

            func  = create_matmul(current_parent,W_var)
        else:
            raise Exception(str("Unrecognized layer function (%s)"%(function_name)))

    layer_type = layer_info["type"]
    if layer_type == "pool":
        strides=layer_info["strides"]
        ksize=layer_info["ksize"]

        cur_layer = create_pool2(current_parent,strides,ksize)
    elif layer_type == "relu":
        if not "function" in layer_info:
            raise Exception("RELU Layer expects a function, none were provided")
        try:
            B_val = layer_info["B_val"]
            B_shape = layer_info["B_shape"]
            B_var = create_B(B_val,B_shape)
        except:
            raise Exception("Invalid/Missing values for relu bias variable")
        #Assumes function assigned above
        cur_layer = tf.nn.relu(func + B_var)
    elif layer_type == "reshape":
        shape = layer_info["shape"]
        cur_layer = tf.reshape(current_parent, shape)
    elif layer_type == "matmul":

        W_shape = layer_info["W_shape"]
        W_std   = layer_info["W_std"]
        W_var = create_W(W_shape,W_std)

        func  = create_matmul(current_parent,W_var)

        try:
            B_val = layer_info["B_val"]
            B_shape = layer_info["B_shape"]
            B_var = create_B(B_val,B_shape)
        except:
            raise Exception("Invalid/Missing values for relu bias variable")

        cur_layer = create_matmul(current_parent,W_var) + B_var
    else:
        raise Exception(str("Unrecognized layer type (%s)"%(layer_type)))
    return cur_layer

'''
'''
This is the example network used. It was taught to play pong, and does not perform well on snake,
but it served as a good baseline

def asri_create_network(self):
    def weight_variable(shape,std):
        initial = tf.truncated_normal(shape, stddev = std)
        return tf.Variable(initial)

    def bias_variable(val,shape):
        initial = tf.constant(val, shape = shape)
        return tf.Variable(initial)

    def conv2d(parent, W_mat, strides):
        return tf.nn.conv2d(parent, W_mat, strides = strides, padding = "SAME")

    def max_pool_2x2(parent,strides,kernel):
        return tf.nn.max_pool(parent, strides = strides,ksize = kernel, padding = "SAME")
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32],.01)
    b_conv1 = bias_variable(.01,[32])

    W_conv2 = weight_variable([4, 4, 32, 64],.01)
    b_conv2 = bias_variable(.01,[64])

    W_conv3 = weight_variable([3, 3, 64, 64],.01)
    b_conv3 = bias_variable(.01,[64])
    
    W_fc1 = weight_variable([1600, 512],.01)
    b_fc1 = bias_variable(.01,[512])

    W_fc2 = weight_variable([512, ACTIONS],.01)
    b_fc2 = bias_variable(.01,[ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, [1, 4, 4, 1]) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1,[1, 2, 2, 1],[1, 2, 2, 1])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1, 2, 2, 1]) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2,[1, 2, 2, 1],[1, 2, 2, 1])

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, [1, 1, 1, 1]) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3,[1, 2, 2, 1],[1, 2, 2, 1])

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1
'''
'''
    def asri_train_network(self,env,s,readout,h_fc1,sess):
        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # open up a game state to communicate with emulator
        game_state = env
        game_state.reset()

        # store the previous observations in replay memory
        D = deque()

        # printing
        a_file = open("logs_" + GAME + "/readout.txt", 'w')
        h_file = open("logs_" + GAME + "/hidden.txt", 'w')

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        r_0, terminal = [0,False]
        x_t = game_state.get_image()
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

        epsilon = INITIAL_EPSILON
        t = 0
        num_finished_eps=0
        start_time = time.time()
        while "pigs" != "fly":
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            for i in range(0, K):
                # run the selected action and observe next state and reward
                r_t, terminal = game_state.step(action_index)
                x_t1 = game_state.get_image()
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

                # store the transition in D
                D.append((s_t, a_t, r_t, s_t1, terminal))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    # if terminal only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch})
            if terminal:
                num_finished_eps+=1
                print(num_finished_eps,r_t)
                if num_finished_eps%1000==0:
                    game_state.reset(render=True)
                else:
                    game_state.reset(render=True)
            

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = 1)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            if t % 500 == 0:
                end_time=time.time()
                steps_per_second = 500.0 / (end_time-start_time)
                print "TIMESTEP", t, "/ Steps/s", steps_per_second, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)
                start_time = time.time()
            # write info to files
            
            #if t % 10000 <= 100:
            #    a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #    h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #    cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            
'''