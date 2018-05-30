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

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt


##DEBUGGING
from math import isnan



#Assumes game_env has the following funtions:
#step(control) Accepts an integer 0->#controls-1, returns (score,done)
#reset(render=False,save=False) (resets the environment), sets the render variable(For final display)
#get_image() gets a numpy array of the game image
#get_image_dims() #Returns dimensions of game screen
#get_num_controls() #Returns number of controls

#CNN_struct

class deepQ():
    #[80,80],4,4,
    def __init__(self,game_env,caffe_file,cost_struct,num_actions,n_episodes=100,e_init=.5,e_final=.05,momentum=.99,
        learning_rate=.001,n_obs_timesteps=500,n_explore_timesteps=500,n_prev_states=590000,minibatch_size=32,
        frames_per_action=1,game_name="name_unspecified",preprocess_func=None):
        self.env = game_env
        self.num_actions = num_actions
        self.n_episodes = n_episodes
        self.e_init = e_init
        self.e_final = e_final
        self.momentum = momentum
        self.learning_rate=learning_rate
        self.n_obs_timesteps = n_obs_timesteps
        self.n_explore_timesteps = n_explore_timesteps
        self.n_prev_states = n_prev_states
        self.minibatch_size = minibatch_size
        self.frames_per_action = frames_per_action
        self.game_name = game_name.replace(" ","_")
        self.preprocess_func = preprocess_func
        build_success = True
        try:
            
            self.game_name,self.online_input,self.online_output,self.online_readout,self.online_vars = \
                build_CNN("q_networks/online",caffe_file=caffe_file)
            self.game_name,self.target_input,self.target_output,self.target_readout,self.target_vars = \
                build_CNN("q_networks/target",caffe_file=caffe_file,input = self.online_input)
                #self.build_CNN_old(board_size,n_board_frames,num_actions,CNN_struct,cost_struct)
            
            
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
            build_success = False
        if build_success:
            try:
                #TODO: Consider creating cost function builder
                # define the cost function
                self.n_outputs = self.target_output.shape[1]
                with tf.variable_scope("train") as scope:
                    #NOTE: This cost function generation is taken from
                    #https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
                    self.X_action = tf.placeholder(tf.int32, shape=[None])
                    self.y = tf.placeholder(tf.float32, shape=[None, 1])
                    q_value = tf.reduce_sum(self.online_output * tf.one_hot(self.X_action, self.n_outputs),
                                            axis=1, keepdims=True)
                    error = tf.abs(self.y - q_value)
                    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
                    linear_error = 2 * (error - clipped_error)
                    self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

                    self.global_step = tf.Variable(0, trainable=False, name='global_step')
                    optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
                    self.training_op = optimizer.minimize(self.loss, global_step=self.global_step)

                print("FINISHED BUILD")
            except Exception as ex_val:
                error_message = str(ex_val)
                print(  "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=\n" + \
                        "Error during cost function generation: (%s)\n"%(error_message) + \
                        "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=")
                build_success = False


        if build_success:
            print("CNN built succesfully")
            print("Input shape = %s"%(self.online_input.shape))
            print("Output shape = %s"%(self.online_output.shape))
            #print("Readout shape = %s"%(self.online_readout.shape))

        else:
            raise Exception("Error during CNN generation, see above messages")
    def train(self,run_test=False):
        self.env.reset()
        input_layer_shape = self.online_input.shape
        self.n_board_frames = input_layer_shape[3]
        single_board_frame_shape = [input_layer_shape[1],input_layer_shape[2],1]
        n_board_frames_less_1 = self.n_board_frames-1

        self.session = tf.InteractiveSession()
        #writer = tf.summary.FileWriter("./tf_log.log", graph=self.session.graph)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        frames = []
        n_max_steps = 10000

        n_steps = 8000000  # total number of training steps
        training_start = 10000  # start training after 10,000 game iterations
        training_interval = 4  # run a training step every 4 game iterations
        save_steps = 1000  # save the model every 1,000 training steps
        copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
        discount_rate = 0.99
        skip_start = 90  # Skip the start of every game (it's just waiting time).
        batch_size = 50
        iteration = 0  # game iterations
        checkpoint_path = str("./saved_networks/%s/%s.ckpt"%(self.game_name,self.game_name))
        done = True # env needs to be reset

        loss_val = np.infty
        game_length = 0
        total_max_q = 0
        mean_max_q = 0.0

        if run_test:
            print("Running test")
            frames = []
            n_max_steps = 30000
            with tf.Session() as sess:
                saver.restore(sess, checkpoint_path)

                obs = env.reset()
                for step in range(n_max_steps):
                    state = self.preprocess_func(obs)

                    # Online DQN evaluates what to do
                    q_values = self.online_output.eval(feed_dict={self.online_input: [state]})
                    print(q_values)
                    action = np.argmax(q_values)
                    print(action)

                    # Online DQN plays
                    obs, reward, done, info = env.step(action)

                    img = env.render(mode="rgb_array")
                    frames.append(img)
                    if done:
                        break
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
            plot_animation(frames)
            plt.show()
            return
        prev_frames = deque(maxlen=self.n_prev_states)

        eps_min = 0.1
        eps_max = 1.0
        eps_decay_steps = 2000000

        def epsilon_greedy(q_values, step_in):
            epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step_in/eps_decay_steps)
            if np.random.rand() < epsilon:
                return np.random.randint(self.n_outputs) # random action
            else:
                return np.argmax(q_values) # optimal action

        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(self.session, checkpoint_path)
            self.copy_online_to_target.run()
        else:
            init.run()
            self.copy_online_to_target.run()

        start_time = time.time()
        while True:
            step = self.global_step.eval()
            if step >= n_steps:
                break
            iteration += 1
            tot_time = time.time()-start_time
            print("                                                                   \r"+"Iteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:2f}  ".format(
                iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
            start_time = time.time()
            if done: # game over, start again
                obs = self.env.reset()
                for skip in range(skip_start): # skip the start of each game
                    obs, reward, done, info = self.env.step(0)
                state = preprocess_func(obs)

            # Online DQN evaluates what to do
            #print("\nINPUT SHAPE",self.online_input.shape)
            #print("PROCESS_SHAPE",state.shape)
            q_values = self.online_output.eval(feed_dict={self.online_input: [state]})
            #print("q_VALUES",q_values)
            action = epsilon_greedy(q_values, step)
            #if np.min(q_values) < .00001:
            #    print("QS")
            #    print(q_values)
            #    print(np.min(state))
            #    print(np.max(state))
            #    return

            # Online DQN plays
            obs, reward, done, info = self.env.step(action)
            next_state = preprocess_func(obs)
            # Let's memorize what happened
            prev_frames.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if (self.global_step.eval() < training_start and iteration < training_start) or iteration % training_interval != 0:
                continue # only train after warmup period and at regular intervals
            
            # Sample memories and use the target DQN to produce the target Q-Value
            #X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            #    sample_memories(batch_size))\

            X_state_val, X_action_val, rewards, X_next_state_val, continues = \
                random.sample(prev_frames, 1)[0]
            print("REWARDS", X_action_val,rewards,continues,len(prev_frames))
            next_q_values = self.target_output.eval(
                feed_dict={self.target_input: [X_next_state_val]})
            max_next_q_values = np.max(next_q_values, axis=1)
            y_val = rewards + continues * discount_rate * max_next_q_values
            #y_val = y_val[0]
            #for i in range(5):
            #    print("HERE2")
            # Train the online DQN
            #print("VALS")
            #print(rewards)
            #print(continues)
            #print("VALS2")
            #print(X_action_val)
            #print(y_val)
            #print(q_values)
            #print("NEXT")
            #print(next_q_values)
            #print(X_action_val)
            #print(max_next_q_values)

            _, loss_val = self.session.run([self.training_op, self.loss], feed_dict={
                self.online_input: np.expand_dims(X_state_val,0), self.X_action: np.array([X_action_val]), self.y: np.array([y_val])})
            #print("VALS")
            #print(rewards)
            #print(continues)
            print("VALS2")
            print(X_action_val)
            print(y_val)
            print(q_values)
            print(loss_val)
            #print("NEXT")
            #print(next_q_values)
            #print(X_action_val)
            #print(max_next_q_values)
            
            if isnan(loss_val) or loss_val > 8e8:
                print(loss_val)
                print(np.expand_dims(X_state_val,0).shape)
                print(X_action_val)
                print(y_val)
                if isnan(loss_val):
                    print("LOSS NAN")
                    return
            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                self.copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(self.session, checkpoint_path)
        
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
                    return img.reshape(88, 80,1)
            
            env = gym.make(game_type)
            num_actions = 9     
    cost_struct = []
    test=False
    if len(sys.argv) > 3:
        test = int(sys.argv[3])>0

    network = deepQ(env,proto_file,cost_struct,num_actions,game_name=game_type,preprocess_func=preprocess_func)
    network.train(run_test=test)