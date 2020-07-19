import dataset #my file
import vae #my file
import lstm_validation #my file
import parameters #my file

class learned_env():

    def __init__(self,action_space):
        self.vae = vae.VAE(dataset)
        self.lstm = lstm_validation.LSTM(action_space, seq_len = 2, batch_size = 1) #create new lstm, maintain the same struct of the lstm used for training (to load pretrained model) but ignore the targets data 


        self.vae.load_json()
        self.lstm.load_json()
        self.current_state = self.lstm.sess.run(self.lstm.zero_state) #initialize current_state to the initial state
        self.restart = 0
        self.frame_count = None
        self.current_frame = self.z_to_img(self.current_state)

    def reset(self):
        self.current_state = self.lstm.sess.run(self.lstm.zero_state)

    # Using the temperature parameter to put stochasticity in lstm predictions
    def get_mix_coef(self,log_mix_coef):
        logmix2 = np.copy(log_mix_coef)/self.temperature 
        logmix2 -= logmix2.max() #normilize
        logmix2 = np.exp(logmix2) #inverse of log
        logmix2 /= logmix2.sum(axis=1).reshape(parameters.LATENT_SIZE, 1)
        return logmix2  

    # Using MDN parameters generate new frame
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
        next_z = chosen

    def step(self,enc_state,act,done_flag):
        
        prev_z = np.zeros((1, 1, parameters.LATENT_SIZE))
        prev_z[0][0] = enc_state

        prev_action = np.zeros((1, 1))
        prev_action[0] = act
    
        prev_restart = np.ones((1, 1))
        prev_restart[0] = done_flag
        
        feed = {
            self.lstm.input_obs: prev_z,
            self.lstm.input_action: prev_action,
            self.lstm.input_res_flag: prev_restart, 
            self.lstm.initial_state: self.current_state
            #self.memory.initial_state: self.memory.sess.run(self.memory.zero_state)
        }

        [log_mix_coef, mean, logstd, reward, predicted_restart_flag, self.current_state] = self.lstm.sess.run([self.lstm.log_mix_coef,self.lstm.mean,self.lstm.logstd,self.lstm.predicted_rewards,self.lstm.predicted_restart_flag,self.lstm.next_state],feed)    

        new_z = self.sample_new_z(log_mix_coef,mean,logstd)

        done = predicted_restart_flag > 0

        return new_z,reward,done


    # Decode 
    def z_to_img(self,z):
        decoded = self.vae.decode_latent_vec(z)
        reconstructed_img = self.vae.post_process_frame(decoded)
        #decoded = self.vae.post_process_frame(decoded)
        return reconstructed_img