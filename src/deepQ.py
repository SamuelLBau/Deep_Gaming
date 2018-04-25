import numpy
from snake_game import snake_game

#Assumes game_env has the following funtions:
#step(control) Accepts an integer 0->#controls-1, returns (score,done)
#reset(render=False,save=False) (resets the environment), sets the render variable(For final display)
#get_image() gets a numpy array of the game image
#get_image_dims() #Returns dimensions of game screen
#get_num_controls() #Returns number of controls
class deepQ():

    def __init__(self,game_env):
        self.env = game_env
    def train(self,n_iter=100):
        #This example runs many iterations
        for n in xrange(n_iter):
            self.env.reset()
            for i in xrange(30):
                self.env.step(3)
    def run(self):
        pass

if __name__ == "__main__":
    env = snake_game()
    network = deepQ(env)
    network.train()

