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
#info_env = get_info_from_env(env)
dataset = dataset.Dataset(env) 
vae = vae.VAE(dataset)
vae.load_json()

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

lstm = lstm.LSTM()
lstm.load_json()
lstm.train_lstm_mdn(batch_encoded_frames,batch_actions,batch_reset)

# create my env
learned_env = lstm_doom_env.DOOM_LSTM_ENV(vae)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

i = 0
while i < 1:
    new_state,reward,done = learned_env.step(batch_encoded_frames[0][0][0],batch_actions[0][0][0],batch_reset[0][0][0])
    new_state = batch_encoded_frames[0][0][0]
    decoded_state = vae.decode_latent_vec(new_state)
    new_frame = vae.post_process_frame(decoded_state)
    imgplot = plt.imshow(new_frame)
    Image.fromarray(new_frame).show()
    '''
    print("mix_coef: ", log_mix_coef)
    for c in range(len(log_mix_coef)):
        print("true z: ",batch_encoded_frames[0][0][0][c])
        print("idx: ", np.where(log_mix_coef[c] == max(log_mix_coef[c])))
        print("my z: ",new_state[c])
    '''

    i += 1
