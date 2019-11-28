import numpy as np
import tensorflow as tf
import gym
from PIL import Image as PilImage
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import pickle
import os
from scipy.misc import imresize as resize
import parameters #my file


class Dataset(object):
    
    def __init__(self,env):
        self.env = env
        self.dataset_size = parameters.DATASET_SIZE
        self.frame_shape = parameters.FRAME_SHAPE
        
        self.vae_batch_size = parameters.VAE_BATCH_SIZE
        self.vae_num_batches = parameters.VAE_NUM_BATCHES
        self.lstm_batch_size = parameters.LSTM_BATCH_SIZE
        self.chunck_size = parameters.SEQ_LENGTH
        
        self.frame_dataset = None
        self.action_dataset = None
        self.reset_dataset = None
        self.encoded_frame_dataset = None

    #Crop image before store to dataset 
    #same preprocessing from orginal world_model source
    def preprocess_frame(self,obs):
        obs = np.array(obs[0:400, :, :]).astype(np.float)/255.0
        obs = np.array(resize(obs, (self.frame_shape[0], self.frame_shape[1])))
        obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        return obs
        

    def create_new_dataset(self, render = True):
        print("Dumping ", self.dataset_size, " frames!")
        frames = []
        actions = []
        resets_env = [] #flag for restarting env after the end of episode

        obs = self.env.reset()
        for itr in range(self.dataset_size):
            if render:
                self.env.render()
            action = self.env.action_space.sample()
            obs,rew,done,_ = self.env.step(action) # take a random action 
            
            actions.append(action)
            frames.append(self.preprocess_frame(obs))
            if done:
                self.env.reset()
                resets_env.append(1)
            else:
                resets_env.append(0)
            
            if itr % 1000 == 0:
                print("Dumped ", itr, " frames!")
        
        #REMEMBER: not finish yet with env. This is a temporary line
        self.env.close() 

        #from len = 1000 to shape = (1000, 64, 64, 3)
        self.frame_dataset = np.stack(frames)
        self.action_dataset = np.stack(actions)
        self.reset_dataset = np.stack(resets_env)


    def store_encoded_dataset(self, frame_only = False, complete= True):
        #dump dataset into folder
        if frame_only:
            np.savez_compressed("dataset/frame_only.npz", frame=self.frame_dataset)
        else:
            if complete:
                assert self.encoded_frame_dataset is not None, "Encode frames first!"
                np.savez_compressed("dataset/complete_dataset.npz", encode=self.encoded_frame_dataset, frame=self.frame_dataset, action=self.action_dataset, reset=self.reset_dataset)        
                print("Encoded Frame Dataset size: ",self.encoded_frame_dataset.shape)
            else:
                assert self.encoded_frame_dataset is not None, "Encode frames first!"
                np.savez_compressed("dataset/dataset.npz", encode=self.encoded_frame_dataset, action=self.action_dataset, reset=self.reset_dataset)
                
            print("Dataset Frames size: ", self.frame_dataset.shape)
            print("Action Dataset size: ",self.action_dataset.shape)
            print("Reset Dataset size: ",self.reset_dataset.shape)

        
    def load_only_frame(self):
        # load preprocessed data
        data = np.load("frame_only/frames.npz")
        self.frame_dataset = data["frame"]
        print("Frame Dataset size: ",self.frame_dataset.shape)
    

    def load_dataset(self, complete = False):
        #dataset_keys = list(raw_data.keys())  #could be usefull
        
        if complete: 
            # load preprocessed data
            raw_data = np.load("dataset/complete_dataset.npz")
            self.frame_dataset = raw_data["frame"] #frame is present only in complete_dataset
        else: 
            raw_data = np.load("dataset/dataset.npz")

        #the other are always present
        self.encoded_frame_dataset = raw_data["encode"]
        self.action_dataset = raw_data["action"]
        self.reset_dataset = raw_data["reset"]


        # reduce loaded dataset to dataset size_
        if self.dataset_size < len(self.encoded_frame_dataset):
            self.encoded_frame_dataset = self.encoded_frame_dataset[0:self.dataset_size]
            self.action_dataset = self.action_dataset[0:self.dataset_size]
            self.reset_dataset = self.reset_dataset[0:self.dataset_size]
        elif self.dataset_size > len(self.encoded_frame_dataset):
            self.dataset_size = len(self.encoded_frame_dataset)

        print("len(self.encoded_frame_dataset): ", len(self.encoded_frame_dataset))
        
        if self.frame_dataset is not None:
            print("Frames Dataset size: ",self.frame_dataset.shape)
        print("Encoded Frame Dataset size: ",self.encoded_frame_dataset.shape)
        print("Action Dataset size: ",self.action_dataset.shape)
        print("Reset Dataset size: ",self.reset_dataset.shape)
        self.num_batches = int(self.dataset_size/(self.lstm_batch_size * parameters.SEQ_LENGTH)) 
        print("Lstm Batch Size : ", self.lstm_batch_size)
        print("Num batches: ", self.num_batches)

  
    
    def get_frames_batches(self):
        assert self.frame_dataset is not None, "Frame dataset is empty!"

        batches = []
        np.random.shuffle(self.frame_dataset)
        for idx in range(self.vae_num_batches):
            data = self.frame_dataset[idx * self.vae_batch_size:(idx+1) * self.vae_batch_size]
            batches.append(data.astype(np.float)/255.0)
        return batches

  
    # chunk  = sequences of encoded_frames,actions,reset_flag each one long parameters.SEQ_LENGTH
    # split data into chunks and shuffle. Before shuffling combine encoded_frame dataset with the others  
    # collect chunks into batches each one long parameters.LSTM_BATCH_SIZE, and shuffle
    # split the combined dataset into 3 separated dataset (one for encoded_frame, one for action and one for reset_flags)
    # return the 3 dataset, splitted in batches of chunks
    def split_dataset_into_batches(self):

        dataset_batched_encoded_frames = [] #shape(num_batches,batch_size,chunk_size,latent_v_size)
        dataset_batched_act = [] #shape(num_batches,batch_size,chunk_size)
        dataset_batched_reset = [] #shape(num_batches,batch_size,chunk_size)

        num_chunks = int(self.dataset_size / self.chunck_size)
        chunks = []

        num_batches = int(num_chunks/self.lstm_batch_size)
        batches = []

        #split dataset in chunks (sequences of frames to train the lstml network)
        for idx in range(num_chunks):
            chuck_enc_frame = self.encoded_frame_dataset[idx * self.chunck_size:(idx+1) * self.chunck_size]
            chunck_act = self.action_dataset[idx * self.chunck_size:(idx+1) * self.chunck_size]
            chunck_reset = self.reset_dataset[idx * self.chunck_size:(idx+1) * self.chunck_size]
            chunks.append([chuck_enc_frame,chunck_act,chunck_reset])

        #break correlataion between different chunks (mantain that in the internal chunk)
        np.random.shuffle(chunks)

        #split list of chunks into batch
        for idx in range(num_batches):
            batches.append(chunks[idx*self.lstm_batch_size:(idx+1)*self.lstm_batch_size])
        
        #shuffle dei batches
        np.random.shuffle(batches)
        
        #split alla dataset in 3 different datasets
        for batch in batches:#10
            batch_encoded_frames = []
            batch_actions = []
            batch_reset = []

            for chunk in batch:#100
                batch_encoded_frames.append(chunk[0])
                batch_actions.append(chunk[1])
                batch_reset.append(chunk[2])

            dataset_batched_encoded_frames.append(batch_encoded_frames)
            dataset_batched_act.append(batch_actions)
            dataset_batched_reset.append(batch_reset)
            
        dataset_batched_encoded_frames = np.array(dataset_batched_encoded_frames)
        dataset_batched_act = np.array(dataset_batched_act)
        dataset_batched_reset = np.array(dataset_batched_reset)

        return dataset_batched_encoded_frames,dataset_batched_act,dataset_batched_reset