import numpy as np
import gym
import matplotlib.pyplot as plt
import vizdoomgym

import dataset #my file
import vae #my file
import lstm #my file
import parameters #my file

'''
before run install vizdoom:
    git clone https://github.com/simontudo/vizdoomgym.git
    cd vizdoomgym
    pip install -e .
'''

#Tensorflow version: 1.14.0


#Hyperparameters
#extract.py that will extract 200 episodes from a random poilcyextract.py that will extract 200 episodes from a random poilcy for 64 times

# Init
env = gym.make('VizdoomTakeCover-v0')
dataset = dataset.Dataset(env) 
vae = vae.VAE(dataset)

#print("Creating dataset!")
#dataset.create_new_dataset(render = False)

print("Loading dataset!")
dataset.load_dataset(complete = False)

#print("Encoding dataset!")
#vae.encode_dataset()

#print("Storing dataset!")
#dataset.store_encoded_dataset(frame_only=True, complete=False)

#get batches from dataset 
batch_encoded_frames,batch_actions,batch_reset = dataset.split_dataset_into_batches()

lstm = lstm.LSTM(dataset)