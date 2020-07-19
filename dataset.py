import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import pickle
import os
import parameters #my file
import PIL.Image as PilImage
from PIL import Image
import subprocess


class Dataset(object):
    
    def __init__(self,env):
        self.env = env
        self.dataset_size = parameters.DATASET_SIZE
        self.frame_shape = parameters.FRAME_SHAPE
        
        self.vae_batch_size = parameters.VAE_BATCH_SIZE
        self.lstm_batch_size = parameters.LSTM_BATCH_SIZE
        self.chunk_size = parameters.SEQ_LENGTH 

        self.dataset = dict() #["obs"],["enc_obs"],["act"],["rew"],["reset"]
        
        

### Creation Dataset 

    #Crop image before store to dataset 
    #same preprocessing from orginal world_model source
    def preprocess_frame(self,obs):       
        #obs = PilImage.fromarray(obs).resize((self.frame_shape[0],self.frame_shape[1]),0) #other library do no works 
        #obs = (255-np.array(obs)) #do not why by applied a negative effects works better FARE CONFRONTO TRA NEGATIVO E NON NEGATIVO 
        
        return obs
        

    def create_new_dataset(self):
        subprocess.call([ "rm", "-rf", "frames"])
        subprocess.call([ "mkdir", "-p", "frames"])

        print("Dumping ", self.dataset_size, " frames!")
        frames = []
        actions = []
        rewards = []
               
        #flag for restarting env after the end of episode
        resets_env = [] 

        #keep time step count to be able to stop otherwise infinitely long running task
        time_step_counter = 0

        action_spec = self.env.action_spec()
        time_step = self.env.reset()
        total_reward = 0

        while time_step_counter < self.dataset_size:
            #frames.append(self.preprocess_frame(time_step.observation['pixels'])) #load initial obs
            obs = self.env.physics.render(height=64, width=64, camera_id=0)
            frames.append(obs)
            
            img = Image.fromarray(obs, 'RGB')
            img.save("frames/frame-%.10d.png" % time_step_counter)

            # take a random action
            action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size=action_spec.shape)

            time_step = self.env.step(action)
            #obs = time_step.observation['pixels']
            
            time_step_counter += 1
            
            
            # Store agent experience
            actions.append(action)
            rewards.append(time_step.reward)
            total_reward += time_step.reward

            if time_step.last():
                resets_env.append(1)
                
                #create a fake final transition to break correlation between final episode and init new episode
                frames.append(obs)
                actions.append(action)
                rewards.append(time_step.reward)
                resets_env.append(1)

                img = Image.fromarray(obs, 'RGB')
                img.save("frames/frame-%.10d.png" % time_step_counter)

                time_step_counter += 1 #to remain in the limit of datateset_size's legth  

                time_step = self.env.reset() #restart env to produce new iterations
            else:
                resets_env.append(0)
            
            if time_step_counter % 1000 == 0:
                print("Dumped ", time_step_counter, " frames!")
        
        subprocess.call(["ffmpeg", "-framerate", "50", "-y", "-i", "frames/frame-%010d.png", "-r", "30", "-pix_fmt", "yuv420p", "dumped_video.mp4"])
        
        #REMEMBER: not finish yet with env. This is a temporary line
        self.env.close()

        print("Total cumulative Reward: ", total_reward)
        self.split_dataset(np.stack(frames),np.stack(actions),np.stack(rewards),np.stack(resets_env))

    
    # slit data into sequences (chunks), shuffle them and split data in train/validation/test sets
    def split_dataset(self,frames,actions,rewards,reset_flags):
        print("len(frames) == len(actions) == len(reset_flags) == len(rewards) : ",len(frames),len(actions),len(reset_flags),len(rewards))
        
        
        assert len(frames) == len(actions) == len(reset_flags) == len(rewards),"in split_dataset the data's length don't match"
        assert len(frames) ==  self.dataset_size, "in split_dataset the data's lenght do not match with the dataset_size"

        #split data in sequences
        num_chunks = int(self.dataset_size / self.chunk_size)
        chunks = []


        #split dataset in chunks (sequences of transizions to train the lstml network)
        for idx in range(num_chunks):
            chunk_frames = frames[idx * self.chunk_size:(idx+1) * self.chunk_size]
            chunk_act = actions[idx * self.chunk_size:(idx+1) * self.chunk_size]
            chunk_rew = rewards[idx * self.chunk_size:(idx+1) * self.chunk_size]
            chunk_reset = reset_flags[idx * self.chunk_size:(idx+1) * self.chunk_size]
            chunks.append([chunk_frames, chunk_act, chunk_rew, chunk_reset])
            
        #break correlataion between different chunks (mantain it in the internal chunk)
        np.random.shuffle(chunks)

        frames = []
        actions = []
        rewards = []
        reset_flags = []


        for chunk in chunks: 
            frames.append(np.array(chunk[0])) # 0 stand for cunck_frames -> [chunk_frames, chunk_act,chunk_rew, chunk_reset] 
            actions.append(np.array(chunk[1]))
            rewards.append(np.array(chunk[2]))
            reset_flags.append(np.array(chunk[3]))
            idx += 1

         # Convert list to np.array
        self.dataset["obs"] = np.stack(frames)
        self.dataset["act"] = np.stack(actions)
        self.dataset["rew"] = np.stack(rewards)
        self.dataset["reset"] = np.stack(reset_flags)

    
    
####### Operations of storing and loading

    def store_dataset(self):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")

        #dump dataset into folder
        np.savez_compressed("dataset/transitions.npz", actions=self.dataset["act"], rewards=self.dataset["rew"], reset_flags=self.dataset["reset"])

        self.store_single_dataset("dataset/obs.npz",self.dataset["obs"])    
            
                
                
    def store_single_dataset(self, dataset_name, data):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")

        np.savez_compressed(dataset_name, frames=data)

    
    def load_encoded_frames(self):
        if not os.path.exists("dataset"):
            raise ValueError('Encoded_frames dataset not found. Impossible to load it')
        
        raw_data = np.load("dataset/enc_obs.npz")
        self.dataset["enc_obs"] = raw_data["frames"] 
        
        
    def load_only_frame(self):
        if not os.path.exists("dataset"):
            raise ValueError('Frame_only dataset not found. Impossible to load it')

        # load preprocessed data
        raw_data = np.load("dataset/obs.npz") # to show the content inside data use -> data.files
        self.dataset["obs"] = raw_data['frames']
        print("Dataset of observations is loaded: ", self.dataset["obs"].shape)
        
    def load_dataset(self):
        #dataset_keys = list(raw_data.keys())  #could be usefull
        if not os.path.exists("dataset"):
            raise ValueError('Frame_only dataset not found. Impossible to load it')
        
        raw_data = np.load("dataset/transitions.npz")
        self.dataset["act"] = raw_data['actions']  
        self.dataset["rew"] = raw_data['rewards']
        self.dataset["reset"] = raw_data['reset_flags']

        self.load_only_frame()
        if os.path.exists("dataset/enc_obs.npz"):
            self.load_encoded_frames()

        print("Loaded current datasets: ", self.dataset.keys())



### Preprocess dataset for VAE training
    
    def split_set_for_vae(self, set_type):
        data = self.dataset["obs"]

        # reshape frames (remove the sequences created for the lstm)
        vae_set = data.reshape(-1,data.shape[2],data.shape[3],data.shape[4]) #mantain the final shape but collapse the first 2
        
        np.random.shuffle(vae_set)

        test_set_size = parameters.TEST_SET_SIZE
        validation_set_size = parameters.VAL_SET_SIZE
        train_set_size = len(data) - test_set_size - validation_set_size

        if set_type == "train_set":
            return self.get_frames_batches(vae_set[0:train_set_size]) 
        if set_type == "val_set":
            return self.get_frames_batches(vae_set[train_set_size:validation_set_size])
        if set_type == "test_set":
            return self.get_frames_batches(vae_set[(train_set_size+validation_set_size):])

    def get_frames_batches(self, dataset):
        assert dataset is not None, "Train Frame dataset is empty!"
        
        # calculate the number of batches
        vae_num_batches = int(len(dataset)/self.vae_batch_size)
        
        batches = []
        for idx in range(vae_num_batches):
            data = dataset[idx * self.vae_batch_size:(idx+1) * self.vae_batch_size]
            batches.append(data.astype(np.float)/255.0)
        
        return batches



### Preprocess dataset for LSTM training
    def split_set_for_lstm(self, set_type):
        assert self.dataset["enc_obs"] is not None, "The observations in dataset are not encoded yet!"
        
        batch_encoded_frames = []
        batch_actions = []
        batch_rewards = []
        batch_reset = []

        # Split cunks in lstm mini batch
        lstm_num_batches = int( (self.dataset["enc_obs"].shape)[0] / self.lstm_batch_size)

        for idx in range(lstm_num_batches):            
            batch_encoded_frames.append(self.dataset["enc_obs"][idx * self.lstm_batch_size:(idx+1) * self.lstm_batch_size])
            batch_actions.append(self.dataset["act"][idx * self.lstm_batch_size:(idx+1) * self.lstm_batch_size])
            batch_rewards.append(self.dataset["rew"][idx * self.lstm_batch_size:(idx+1) * self.lstm_batch_size])
            batch_reset.append(self.dataset["reset"][idx * self.lstm_batch_size:(idx+1) * self.lstm_batch_size])

        # Split data in train_set,validation_set and test_set
        lstm_test_range = parameters.LSTM_TEST_CHUNKS
        lstm_val_range = parameters.LSTM_VAL_CHUNKS + lstm_test_range
        

        if set_type == "test_set":
            return batch_encoded_frames[0:lstm_test_range],batch_actions[0:lstm_test_range],batch_rewards[0:lstm_test_range],batch_reset[0:lstm_test_range]

        if set_type == "val_set":
            return batch_encoded_frames[lstm_test_range:lstm_val_range],batch_actions[lstm_test_range:lstm_val_range],batch_rewards[lstm_test_range:lstm_val_range],batch_reset[lstm_test_range:lstm_val_range]

        if set_type == "train_set":
            return batch_encoded_frames[lstm_val_range:],batch_actions[lstm_val_range:],batch_rewards[lstm_val_range:],batch_reset[lstm_val_range:]
        
        

       
       
       
