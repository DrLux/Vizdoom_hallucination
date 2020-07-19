import numpy as np
import gym
import matplotlib.pyplot as plt
from dm_control.suite.wrappers import pixels
from dm_control import suite

import dataset #my file
import vae #my file
import parameters #my file
import lstm #my file


from PIL import Image 


    
#Tensorflow version: 1.14.0

# Load environment from the Deepminf control suit
env = suite.load("reacher","easy",visualize_reward=True)

# Wrap env to visualize only frames observations
#env = pixels.Wrapper(feature_vectors_env)
action_space = env.action_spec().shape[0]


#### Create dataset
# Create dataset
dataset = dataset.Dataset(env) 
dataset.load_dataset()
#dataset.create_new_dataset()
#dataset.store_dataset()
#dataset.load_only_frame()

# Create and train VAE
vae = vae.VAE(dataset)
vae.load_json()
#vae.train_vae(checkpoint=True)
#vae.test_vae()
#vae.encode_dataset()



# Create and train LSTM
#lstm = lstm.LSTM(action_space)
#batch_encoded_frames,batch_actions,batch_rewards,batch_reset = dataset.split_set_for_lstm("train_set")
#lstm.train_lstm_mdn(batch_encoded_frames,batch_actions,batch_rewards,batch_reset)




### DEVO ANCORA GESTIRE LA COSA DELLO STATO CLONE CHE MI HA DETTO ANTONIO
batch_encoded_frames,batch_actions,batch_rewards,batch_reset = dataset.split_set_for_lstm("train_set")


#lstm_train = lstm.LSTM(action_space, LSTM_TYPE = "TRAINING") 
#lstm_train.train_lstm_mdn(batch_encoded_frames,batch_actions,batch_rewards,batch_reset)


lstm_inf = lstm.LSTM(action_space, seq_len = 2, batch_size = 1)
lstm_inf.load_json()

#Initialize fist lstm state
lstm_state = lstm_inf.sess.run(lstm_inf.initial_state)

# Peek a random frame into dataset and encode it
idx = 5
frame = dataset.dataset["obs"][1][idx]
encoded_frame = vae.encode_input(frame)

# Reshape placeholders 
prev_z = np.zeros((1, 1, parameters.LATENT_SIZE))
prev_z[0][0] = encoded_frame

prev_action = np.zeros((1, 1,action_space))
prev_action[0] = dataset.dataset["act"][1][idx]


prev_rew = np.zeros((1,1))
prev_rew[0] = dataset.dataset["rew"][1][idx]
prev_rew = prev_rew.astype(np.float32)

prev_restart = np.ones((1, 1))
prev_restart[0] = dataset.dataset["reset"][1][idx]


# Feed placeholders
feed = {
    lstm_inf.input_obs: prev_z,
    lstm_inf.input_action: prev_action,
    lstm_inf.input_rewards: prev_rew,
    lstm_inf.input_res_flag: prev_restart,
    lstm_inf.initial_state: lstm_state,
}

# Do 1-step inference
#new_enc_frame,done,rew,lstm_state = lstm_inf.sess.run([lstm_inf.target_obs,lstm_inf.target_restart,lstm_inf.target_rewards,lstm_inf.next_state], feed)
done,rew = lstm_inf.sess.run([lstm_inf.predicted_restart_flag,lstm_inf.predicted_rewards], feed)


print("Done: ", done)
print("ReW: ", rew)


'''
# Decode producted frames
decoded_frame = vae.decode_latent_vec(new_enc_frame)
reconstructed_frame = np.round(reconstructed_frame * 255.).astype(np.uint8)
reconstructed_frame = reconstructed_frame.reshape(64, 64, 3) #remove batch dimension

# Show results
plt.figure(3)
plt.imshow(reconstructed_frame)
plt.show()

print("Done: ",done, " should be: ", dataset.dataset["res"][1][idx])
print("Reward: ",rew," should be: ", dataset.dataset["rew"][1][idx])
'''