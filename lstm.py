import numpy as np
import parameters #my file

# for tf1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json


class LSTM(object):
    
    def __init__(self,dataset):
        self.latent_size = parameters.LATENT_SIZE
        self.dataset = dataset

        self.num_mixture = parameters.MIXTURE 
        self.latent_size = parameters.LATENT_SIZE
        self.seq_length = parameters.SEQ_LENGTH # assume every sample has same length.
        self.input_size = parameters.INPUT_SIZE    
        self.dim_cell_state = parameters.DIM_CELL_STATE
        self.tiny = parameters.TINY         
        self.learning_rate = parameters.LSTM_LEARNING_RATE
        self.batch_size = parameters.LSTM_BATCH_SIZE

        self.sess = None
        self.batch_enc_frames = None
        self.batch_action = None
        self.batch_restart = None

        #placeholders
        self.batch_obs = None
        self.batch_action = None
        self.batch_restart_flags = None

        self.target_obs = None
        self.target_restart = None

        self.input_seq = None #combine obs,action and batch_restart_flags into one single vector

        #lstm cell
        self.cell = None
        self.initial_state = None
        #self.final_state = None
        self.outputs = None #for each time stamp collect lstm state (c,h) each one with shape (batch_size,lstm_state)
        
        #loss value
        self.z_cost = None
        self.reset_cost = None
        self.total_cost = None
        self.optimizer = None

        with tf.variable_scope('mdn_rnn', reuse=True):
            self.graph = tf.Graph() 
            with self.graph.as_default():
                #init routines
                #self.init_sess()
                self.init_cell()
                self.define_placeholder()
                self.forward_pass()
                predicted_restart_flag,output_for_mdn = self.activation_function_RNN()
                self.mdn(output_for_mdn,predicted_restart_flag)
                #self.get_parameters()


    #this is directly from https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py
    # I didn't find the specific formula online 
    def z_loss_func(self,log_mix_coef,mean,logstd):
        # reshape target data so that it is compatible with prediction shape
        # reshape in ordert to have a lot of vectors with a single dimension
        # from shape=(100, 500, 64) to shape=(3200000, 1)
        flat_target_z = tf.reshape(self.target_obs,[-1, 1])

        print("self.target_obs: ", self.target_obs)
        print("flat_target_z: ", flat_target_z)
        print("mean: ", mean)

        assert 1 == 2, "ciaooo"

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        tf_lognormal = -0.5 * ((flat_target_z - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI
        v = tf.reduce_logsumexp(log_mix_coef + tf_lognormal, 1, keepdims=True)
        return -tf.reduce_mean(v)

    def restart_loss_fun(self,predicted_restart_flag):
        # from shape=(50000,) to shape=(50000,1)
        flat_target_restart = tf.reshape(self.target_restart, [-1, 1])

        r_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_restart,
                                                                logits=tf.reshape(predicted_restart_flag,[-1, 1]))
        # factor of importance for restart=1 rare case for loss
        #factor = tf.ones_like(r_cost) + flat_target_restart * (self.hps.restart_factor-1.0)
        #return tf.reduce_mean(tf.multiply(factor, r_cost))

        return r_cost

    def mdn(self,mdn_input,predicted_restart_flag):
        #mdn_input = # shape=(3200000, 15) 

        #extract the parameters of each of 5 gaussian
        #each one have shape(3200000, 5)
        #PERCHÉ DICE CHE IN OUTPUT HA GIA IL LOGMIX? NON DOVREI IO FARE IL LOG DEL RISULTATO?
        log_mix_coef, mean, logstd = tf.split(mdn_input, 3, 1) 

        #apply softmax function over the mix_coef but not directly, instead use the log trick to avoid multiplications and divisions (and overflow/underflow problems)
        #reduce_logsumexp: https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        log_mix_coef = log_mix_coef - tf.reduce_logsumexp(log_mix_coef, 1, keepdims=True)
        
        self.z_cost = self.z_loss_func(log_mix_coef, mean, logstd)
        self.reset_cost = self.restart_loss_fun(predicted_restart_flag)

        self.total_cost = self.z_cost + self.reset_cost

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_cost)

        #####################
        # Tensorflow RUN
        self.init_sess()
        batch_encoded_frames,batch_actions,batch_reset = self.dataset.split_dataset_into_batches()            
        z_cost,reset_cost,_ = self.sess.run([self.z_cost, self.reset_cost, self.optimizer], {self.batch_obs: batch_encoded_frames[0], self.batch_action: batch_actions[0],self.batch_restart_flags: batch_reset[0]})
        print("z_cost,reset_cost: ", z_cost,reset_cost)
        # Fine Tensorflow RUN
        ##########################

        assert 1 == 2, "ciao"
        

    #The output of rnn is processed with a linear activation function to produce MDN input
    def activation_function_RNN(self):
        # the final output must have the shape of the MDN input -> total = 961
        # every gaussian mixture need 3 parameters (mu,sigma and mixture coefficent) 
        # the last dimension is needed to predict the restart state of the produced state
        SIZE_FINAL_OUTPUT = self.latent_size * (self.num_mixture * 3) + 1 
        

        with tf.variable_scope('Activation_function_RNN'):
            output_w = tf.get_variable("rnn_output_w", [self.dim_cell_state, SIZE_FINAL_OUTPUT])
            output_b = tf.get_variable("rnn_output_b", [SIZE_FINAL_OUTPUT])

        #output: concatenate all batch data for each timestamp into one big vector with length = lstm_state. --> from (500,100,512) to  shape=(50000, 512)
        output = tf.reshape(tf.concat(self.outputs, axis=1), [-1, self.dim_cell_state])
        
        output = tf.reshape(output, [-1, self.dim_cell_state]) #questa riga mi sembra inutile. in teoria trasforma [1,2,3] in [[1,2,3]]. IN pratica non sembra cambiare la shape e se la togli non cambia il risultato 

        #shape:  (50000, 961) -> 50000 = batch_size * seq_len 
        output = tf.nn.xw_plus_b(output, output_w, output_b)
       
        #shape: (50000,)
        predicted_restart_flag = output[:, 0] #the first dimension will take care of restart flags

        #shape:  (50000, 960)
        output_for_mdn = output[:, 1:] #the output values expect the restart_flags in first dimension            
        
        # shape=(3200000, 15): reshape data into a 15dimensional vector (num gaussian mixtrue * parameters to indicate a gaussian mixture)
        output_for_mdn = tf.reshape(output_for_mdn, [-1, self.num_mixture * 3])

        return predicted_restart_flag,output_for_mdn


    def forward_pass(self):
        with tf.variable_scope("rnn_forward_pass"):
            # split input for time_step: create a "self.seq_length" long list of  (self.batch_size, latent_size+act+rest) tensor from input_seq
            #len(inputs.shape(self.batch_size, latent_size+act+rest)) = self.seq_length
            inputs = tf.unstack(self.input_seq, axis=1)

            state = self.initial_state
            zero_c, zero_h = self.initial_state
            outputs = []
            prev = None

            for i in range(self.seq_length):
                if i > 0:
                    #Set the current variable scope to true
                    tf.get_variable_scope().reuse_variables()

                # for current i-th time stamp check for each entry of batch if restart_flag is setted to 1  
                # boolean vector of length "batch_size". Indicates with true if that sequence is intial sequences
                restart_flag = tf.greater(self.batch_restart_flags[:, i], 0.5)
                
                c, h = state

                # where(condition,vector_x,vector_y): assign y[i] to x[i] where condition[i] is true
                # if restart is 1, then reset lstm state to zero
                c = tf.where(restart_flag, zero_c, c)
                h = tf.where(restart_flag, zero_h, h)

                inp = inputs[i]

                    
                # feed current input and the state of previous cell (setted to 0 if the episode is resetted) 
                output, state = self.cell(inp, tf.nn.rnn_cell.LSTMStateTuple(c, h))
                outputs.append(output)
                
            #self.final_state:  LSTMStateTuple(c=tensor(shape=(100, 512)), h=Tensor(shape=(100, 512))
                    #self.final_state = state  per ora non mi serve salvarmelo
            #self.outputs: a list long "seq_len" of lstm states (each one of shape=(batch_size, lstm_state)) 
            self.outputs = outputs


    def init_cell(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(self.dim_cell_state) #init cell with 512 hidden state size
        
        # the state of LSTM is based on two element. C and H each other zero-filled tensors with shape of (batch_size, dim_cell_state) 
        # LSTMStateTuple(c=<tf.Tensor 'LSTMCellZeroState/zeros:0' shape=(100, 512) dtype=float32>, h=<tf.Tensor 'LSTMCellZeroState/zeros_1:0' shape=(100, 512) dtype=float32>)
        self.initial_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        
    def define_placeholder(self):
        self.batch_obs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length, self.latent_size], name="batch_enc_frames")
        self.batch_action = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length], name="batch_action")
        self.batch_restart_flags = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length], name="batch_restart")

        self.target_obs = self.batch_obs[:, 1:, :]
        self.target_restart = self.batch_restart_flags[:, 1:]

        # Concatenates input_obs,input_action_input_restart in one single vector
        #input_seq:  Tensor("concat:0", shape=(100, 999, 66), dtype=float32)
        #self.input_seq = shape=(self.latent_size, self.seq_length, latent_size+act+rest)
        self.input_seq = tf.concat([    self.batch_obs,
                                        tf.reshape(self.batch_action, [self.batch_size, self.seq_length, 1]),
                                        tf.reshape(self.batch_restart_flags, [self.batch_size, self.seq_length, 1])    
                                    ], axis=2)
    
    def init_sess(self):
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init_vars)
        print("Session Started!")

    def get_parameters(self):
        t_vars = tf.trainable_variables()
        for v in t_vars:
            print(v.name) 
