import numpy as np
import gym
import matplotlib.pyplot as plt
import vizdoomgym

import dataset #my file
import vae #my file
import lstm #my file
import parameters #my file
import lstm_doom_env #my file

from PIL import Image 

import pynput
from pynput import keyboard
from pynput.keyboard import Key, Controller

'''
before run install vizdoom:
    git clone https://github.com/simontudo/vizdoomgym.git
    cd vizdoomgym
    pip install -e .
'''

#def get_info_from_env(env):
    
#Tensorflow version: 1.14.0

# Init
env = gym.make('VizdoomTakeCover-v0')

def key_press(key):
    global act
    if key==Key.left:
        act[0] = 0
        print('human key left.')
    if key==Key.right:
        act[0] = 1
        print('human key right.')

def key_release(key):
    global act
    act[0] = 0.5

       
vae = vae.VAE(dataset)
vae.load_json()

lstm = lstm.LSTM()

#lstm.load_json()




#Create and store dataset ###
#print("Creating dataset!")
#dataset.create_new_dataset(render = False)
#vae.encode_dataset()

#print("Storing dataset!")
#dataset.store_encoded_dataset(frame_only=False, complete=True)



# Load dataset
######dataset.load_dataset(complete=True)

#vae.encode_dataset()

#get batches from dataset 
#batch_encoded_frames,batch_actions,batch_reset = dataset.split_dataset_into_batches()

#lstm.train_lstm_mdn(batch_encoded_frames,batch_actions,batch_reset)

# create my env
learned_env = lstm_doom_env.DOOM_LSTM_ENV(vae)

initial_frame = [-0.91315293,  1.2236881,   0.17299916,  0.10731404,  0.5680487,  -0.49503556,
  2.0010276,   1.1171637,   1.3731774,  -0.8313165,   2.2233753,   0.46785936,
  0.15495707,  0.4782607 ,  0.7822306,   0.35097674, -0.75028044, -0.16706006,
  1.3053885 , -1.2359716 , -1.5131272,   0.63023174, -1.9062142 ,  0.00694928,
 -0.714864  , -0.7387922 , -0.3647223,  -0.5580204 , -0.25286108, -0.434293,
  0.7238775 ,  0.40156108, -0.5594823,  -1.2085987 , -0.29880083,  0.6063757,
  0.5483104 ,  0.9401099 , -0.7121125,  -0.46440625, -0.36887395,  0.16015734,
 -1.1010108 , -0.461102  ,  1.4397485,  -0.45861107, -0.29993907, -0.5142053,
  0.09924194,  1.0011657 ,  0.08839466,  1.226267  , -1.2972203,  -1.8790438,
 -0.12851784, -0.20420484, -1.1443287 , -0.00856118, -1.1295127,   0.897585,
  0.8546144 , -1.0315335 ,  0.14407876,  1.5014809]


#initial_frame = dataset.encoded_frame_dataset[0]

input("Press Enter to init session")


from gym.envs.classic_control import rendering
dataset = dataset.Dataset(env) 

done = False
viewer = None
act = np.array([0])
act[0] = 0.8
obs = env.reset()
counter = 0
viewer = None

with keyboard.Listener(
    on_press=key_press,
    on_release=key_release) as listener:

    while counter <= 7000:
        obs,rew,done,_ = env.step(act)
        if done:
            env.reset()
            print("You DIED")
        counter += 1
        if viewer == None:
            viewer = rendering.SimpleImageViewer()
        if counter == 4000:
            input("Press Enter to Crop and resize frames!")
        if counter >= 4000:
            obs = dataset.preprocess_frame(obs)
        viewer.imshow(obs)


input("Press Enter to start the Dream Env!")

learned_env.game(initial_frame)
