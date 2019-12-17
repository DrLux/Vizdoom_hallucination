import dataset #my file
import vae #my file
import lstm #my file
import parameters #my file

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import gym


import pynput
from pynput import keyboard
from pynput.keyboard import Key, Controller

class DOOM_LSTM_ENV(gym.Env):
    
    def __init__(self,vae):
        self.eyesight = vae
        self.seq_length = 2
        self.batch_size = 1
        self.temperature = 0.2 # tplay witrh 1.10
        self.memory = lstm.LSTM(seq_len = self.seq_length, batch_size = self.batch_size) #create new lstm, maintain the same struct of the lstm used for training (to load pretrained model) but ignore the targets data 
        self.memory.load_json()
        self.current_state = self.memory.sess.run(self.memory.zero_state)
        self.restart = 0
        self.frame_count = None
        self.action = np.array([0.5])
        
    def key_press(self,key):#,mod):
        if key==Key.left:
            self.action[0] = -1.0
            print('human key left.')
        if key==Key.right:
            self.action[0] = +1.0
            print('human key right.')

    def key_release(self,key):#, mod):
        self.action[0] = 0.5
            
    def get_mix_coef(self,log_mix_coef):
        logmix2 = np.copy(log_mix_coef)/self.temperature 
        logmix2 -= logmix2.max() #normilize
        logmix2 = np.exp(logmix2) #inverse of log
        logmix2 /= logmix2.sum(axis=1).reshape(parameters.LATENT_SIZE, 1)
        return logmix2        


    def sample_new_z(self,log_mix_coef,mean,logstd):     
        mixture_idx = np.zeros(parameters.LATENT_SIZE)
        chosen_mean = np.zeros(parameters.LATENT_SIZE)
        chosen_logstd = np.zeros(parameters.LATENT_SIZE)

        mix_coef = self.get_mix_coef(log_mix_coef)

        for d in range(parameters.LATENT_SIZE):
            mix_id = np.random.choice(parameters.MIXTURE, 1, p=mix_coef[d]) #samples from a categorial distribution
            mixture_idx[d] = mix_id
            chosen_mean[d] = mean[d][mix_id]
            chosen_logstd[d] = logstd[d][mix_id]

        rand_gaussian = np.random.randn(parameters.LATENT_SIZE)*np.sqrt(self.temperature)
        next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
        return next_z
        
    
    def reset(self):
        self.restart = 0
        self.current_state = self.memory.sess.run(self.memory.zero_state)



    def step(self,enc_state,act,done_flag):
        
        prev_z = np.zeros((1, 1, parameters.LATENT_SIZE))
        prev_z[0][0] = enc_state

        prev_action = np.zeros((1, 1))
        prev_action[0] = act
    
        prev_restart = np.ones((1, 1))
        prev_restart[0] = done_flag
        
        feed = {
            self.memory.input_obs: prev_z,
            self.memory.input_action: prev_action,
            self.memory.input_res_flag: prev_restart, 
            self.memory.initial_state: self.current_state
            #self.memory.initial_state: self.memory.sess.run(self.memory.zero_state)
        }

        [log_mix_coef, mean, logstd, predicted_restart_flag, self.current_state] = self.memory.sess.run([self.memory.log_mix_coef,self.memory.mean,self.memory.logstd,self.memory.predicted_restart_flag,self.memory.next_state],feed)    

        new_z = self.sample_new_z(log_mix_coef,mean,logstd)

        reward = 1
        done = False

        if predicted_restart_flag > 0:
            reward = 0
            done = True

        return new_z,reward,done

    #def generate_initial_state(self):



    def z_to_img(self,z):
        decoded = self.eyesight.decode_latent_vec(z)
        decoded = self.eyesight.post_process_frame(decoded)
        return decoded

    
    def game(self,initial_state):
        done = False
        new_z = initial_state
        viewer = None
        counter = 0
        act = 0

        # Collect events until released
        with keyboard.Listener(
                on_press=self.key_press,
                on_release=self.key_release) as listener:

            while counter <= 4000:
                #act = random.randint(0,1)
                new_z,reward,done = self.step(new_z,self.action[0],done)
                if done:
                    print("You DIED")
                img = self.z_to_img(new_z)
                #img = Image.fromarray(img, 'RGB')
                counter += 1
                if viewer is None:
                    viewer = rendering.SimpleImageViewer()
                viewer.imshow(img)
                #act = input("inserire input")
            