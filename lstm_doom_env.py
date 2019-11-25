import dataset #my file
import vae #my file
import lstm #my file
import parameters #my file

import numpy as np
'''
class DOOM_LSTM_ENV(object):
    def __init__(self,vae):
        self.eyesight = vae
        self.seq_length = 2
        self.batch_size = 1
        self.temperature = 1.25 # train with this temperature

        self.memory = lstm.LSTM(seq_len = self.seq_length, batch_size = self.batch_size) #create new lstm, maintain the same struct of the lstm used for training (to load pretrained model) but ignore the targets data 

    def get_mix_coef(self,log_mix_coef):
        logmix2 = np.copy(log_mix_coef)/self.temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(parameters.LATENT_SIZE, 1)
        return logmix2        

    def sample_new_z(def,log_mix_coef,mean,logstd):     
        mixture_idx = np.zeros(parameters.LATENT_SIZE)
        chosen_mean = np.zeros(parameters.LATENT_SIZE)
        chosen_logstd = np.zeros(parameters.LATENT_SIZE)

        mix_coef = get_mix_coef(log_mix_coef)

        for d in range(parameters.LATENT_SIZE):
            mix_id = 
            mixture_idx[d] = mix_id
            chosen_mean[d] = mean[d][mix_id]
            chosen_logstd[d] = logstd[d][mix_id]

        rand_gaussian = np.random.randn(parameters.LATENT_SIZE)*np.sqrt(temperature)
        next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
        return next_z
    
    # Nome provvisorio
    def get_data(self,enc_state,act,done_flag):

        prev_z = np.zeros((1, 1, parameters.LATENT_SIZE))
        prev_z[0][0] = enc_state

        prev_action = np.zeros((1, 1))
        prev_action[0] = act
    
        prev_restart = np.ones((1, 1))
        prev_restart[0] = done_flag
        
        #I do not feed the lstm state because the class do it for me automatically
        feed = {
            self.memory.input_obs: prev_z,
            self.memory.input_action: prev_action,
            self.memory.input_res_flag: prev_restart, 
        }

        [log_mix_coef, mean, logstd, predicted_restart_flag] = self.memory.sess.run([self.memory.log_mix_coef,self.memory.mean,self.memory.logstd,self.memory.predicted_restart_flag],feed)    
        
        print("log_mix_coef: ", log_mix_coef.shape)
        print("mean: ", mean.shape)
        print("logstd: ", logstd.shape)
        print("predicted_restart_flag: ", predicted_restart_flag.shape)
'''