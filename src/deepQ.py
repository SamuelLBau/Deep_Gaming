import numpy


#Assumes game_env has the following funtions:
#step(control) Accepts an integer 0->#controls-1, returns (score,done)
#reset() (resets the environment)
#get_image() gets a numpy array of the game image
#get_image_dims() #Returns dimensions of game screen
#get_num_controls() #Returns number of controls
class deepQ():

    def __init__(self,game_env):
        self.env = game_env

