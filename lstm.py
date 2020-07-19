import numpy as np
import parameters #my file

# for tf1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json

#for logging
import logging
import psutil
        

class LSTM(object):
    
    def __init__(self,action_space,seq_len = parameters.SEQ_LENGTH, batch_size = parameters.LSTM_BATCH_SIZE, LSTM_TYPE = "INFERENCE"):
        #Type of LSTM
        self.lstm_type = LSTM_TYPE #Indicate the purpose of lstm. The inference one do not need the training stuff
        
        #parameters
        self.seq_length = seq_len -1 # total step are 499, the total batch size is 499+1 (without training total step and total batch size could be the same size). We must leave last timestap for the target. X from 0 to N-1 and Y from 1 to N 
        self.batch_size = batch_size
        self.latent_size = parameters.LATENT_SIZE
        self.num_mixture = parameters.MIXTURE 
        self.latent_size = parameters.LATENT_SIZE
        self.input_size = parameters.INPUT_SIZE    
        self.dim_cell_state = parameters.DIM_CELL_STATE
        self.learning_rate = parameters.LSTM_LEARNING_RATE
        self.global_step = None
        self.action_space = action_space

        #session
        self.sess = None
        self.assign_ops = {} #dictionary to collect assignment operations to reload the model weights
        
        #placeholders
        self.batch_obs = None
        self.batch_action = None
        self.batch_rewards = None
        self.batch_restart_flags = None

        self.input_obs = None #for inference do not need to feed the entire batch but we just need the input
        self.input_action = None
        self.input_res_flag = None
        self.input_rewards = None

        self.target_rewards = None
        self.target_obs = None
        self.target_restart = None

        self.input_seq = None #combine obs,action and batch_restart_flags into one single vector

        #lstm cell
        self.curr_learn_rate = None
        self.cell = None
        self.zero_state = None # to initialize cell to zeros
        self.initial_state = None #the firts step of a sequence
        self.outputs = None #for each time stamp collect lstm state (c,h) each one with shape (batch_size,lstm_state)
        self.predicted_restart_flag = None
        self.predicted_rewards = None
        
        #mdn
        self.log_mix_coef = None
        self.mean = None
        self.logstd = None
        self.actual_learn_rate = None

        #loss value
        self.z_cost         = None
        self.reset_cost     = None
        self.reward_cost    = None
        self.total_cost     = None
        self.optimizer      = None
        
        logging.basicConfig(filename='memory_leak.log',format='%(asctime)s - %(message)s', level=logging.INFO)    
        
        self.graph = tf.Graph() 
        with self.graph.as_default():
            #init routines
            self.init_cell()
            self.define_placeholder()
            self.unroll()
            output_for_mdn = self.activation_function_RNN()
            if self.lstm_type != "INFERENCE":
                self.mdn(output_for_mdn)
            self.init_sess()
            self.collect_assign_ops()  

    
    def log_info(self,lr,zc,rc,rwc,tc,msg):
        mem = dict(psutil.virtual_memory()._asdict())
        #logging.info ('Memory available: {} ({}), learning_rate: {}, z_cost: {}, reset_cost: {}, total_cost: {} + {}'.format(mem["available"],mem["percent"],lr,zc,rc,tc,msg))
        
        '''
        logging.info("#### {} #####".format(msg))
        logging.info('Memory available: {} ({}%) ***'.format(mem["available"],mem["percent"]))
        logging.info('learning_rate: {}'.format(lr))
        logging.info('z_cost: {}'.format(zc))
        logging.info('reset_cost: {}'.format(rc))
        logging.info('total_cost: {} ***'.format(tc))
        logging.info("#########\n")
        '''


        print("#### {} #####".format(msg))
        #print('Memory available: {} ({}%) ***'.format(mem["available"],mem["percent"]))
        print('learning_rate: {}'.format(lr))
        print('z_cost: {}'.format(zc))
        print('reset_cost: {}'.format(rc))
        print('reward_cost: {}'.format(rwc))
        print('\ttotal_cost: {}  ***'.format(tc))
        print("#########\n")

    def train_lstm_mdn(self, batch_encoded_frames,batch_actions,batch_rewards,batch_reset):   
        
        self.graph.finalize() #no more node can be added to the graph (block memory leak)
        file_to_store = 0
        global_step = 0
        curr_learn_rate = self.learning_rate
        record_loss = 10000
        log_loss_value = []

        for epoch in range(1,parameters.LSTM_EPOCH_TRAIN):

            batch_initial_state = self.sess.run(self.initial_state)
            

            for b in range(len(batch_encoded_frames)):

                feed = {
                    self.batch_obs: batch_encoded_frames[b],
                    self.batch_action: batch_actions[b],
                    self.batch_rewards: batch_rewards[b],
                    self.batch_restart_flags: batch_reset[b],
                    self.initial_state: batch_initial_state,
                    self.curr_learn_rate: curr_learn_rate
                }

                z_cost,reset_cost,reward_cost,total_cost,batch_initial_state,_ = self.sess.run([self.z_cost, self.reset_cost, self.reward_cost, self.total_cost,self.next_state,self.optimizer], feed)
                log_loss_value.append(total_cost)

                # Only when achieve a better result save checkpoint
                if total_cost < record_loss:
                    record_loss = total_cost

                    # Is pointless saving so frequently as with the first epochs
                    if epoch >= 3:

                        if epoch%2 == 0:
                            self.save_json()
                        else:
                            self.save_json()
                            self.save_json(jsonfile='models/backup_lstm_model.json')

                        f = open("models/lstm_best_loss_value.txt", "a")
                        f.write("\n"+str(epoch)+" -> "+str(record_loss))
                        f.close()

                
                global_step = self.sess.run(self.inc_global_step)
                #curr_learn_rate = (self.learning_rate - parameters.LSTM_MIN_LEARNING_RATE) * (parameters.LSTM_LEARNING_RATE_DECAY) ** global_step + parameters.LSTM_MIN_LEARNING_RATE

                self.log_info(curr_learn_rate,z_cost,reset_cost,reward_cost,total_cost,str(epoch))

            

        
                

    #this is directly from https://github.com/hardmaru/WorldModelsExperiments/blob/master/doomrnn/doomrnn.py
    def z_loss_func(self):
        #shape=(319360, 1)
        flat_target_z = tf.reshape(self.target_obs, [-1, 1])
        flat_predicted_z = tf.reshape(self.predicted_z, [-1, 1])

        #This is a regression task so MSE
        #z_cost = tf.reduce_mean(tf.reduce_sum(tf.square(flat_target_z - flat_predicted_z)))
        z_cost = tf.reduce_mean(tf.reduce_sum(tf.square(flat_target_z - flat_predicted_z),reduction_indices = [1]))

        return tf.reduce_mean(z_cost)


    def restart_loss_fun(self):
        #shape=(4990,1)
        flat_target_restart = tf.reshape(self.target_restart, [-1, 1])
        flat_predicted_restart = tf.reshape(self.predicted_restart_flag,[-1, 1])

        r_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_restart,logits=flat_predicted_restart)
        return tf.reduce_mean(r_cost)

    def reward_loss_fun(self):
        #shape=(4990,1)
        flat_target_rewards = tf.reshape(self.target_rewards, [-1, 1])
        flat_predicted_rewards = tf.reshape(self.predicted_rewards,[-1, 1])

        rw_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_rewards,logits=flat_predicted_rewards)
        return tf.reduce_mean(rw_cost)
        

    
    def mdn(self,mdn_input):
        self.z_cost = self.z_loss_func()
        self.reset_cost = self.restart_loss_fun()
        self.reward_cost = self.reward_loss_fun()

        self.total_cost = self.z_cost + self.reset_cost + self.reward_cost
        
        self.optimizer = tf.train.AdamOptimizer(self.curr_learn_rate).minimize(self.total_cost)

    #The output of rnn is processed with a linear activation function to produce MDN input
    def activation_function_RNN(self):
        SIZE_FINAL_OUTPUT = 66#10 * 499 * (64 + 2)
        
        output = tf.reshape(tf.concat(self.outputs, axis=1), [-1, self.dim_cell_state])
        
        with tf.variable_scope('lstm_activation_function'):
            output_w = tf.get_variable("rnn_output_w", [self.dim_cell_state, SIZE_FINAL_OUTPUT])
            output_b = tf.get_variable("rnn_output_b", [SIZE_FINAL_OUTPUT])
        
        output = tf.nn.xw_plus_b(output, output_w, output_b)

        
        self.predicted_restart_flag = output[:, 0] # shape=(4990,1) -> the first dimension will take care of restart flags
        self.predicted_rewards = output[:, 1] # shape=(4990,1) -> the secondo dimension will take care of reward
        self.predicted_z = output[:, 2:] # shape=(4990,1) -> the output values expect the restart_flags in first dimension            

    # invece di gestire l'addestramente un frame alla volta lui lo fa tutto insieme, (tutto il batch (vettorializzando) per tutta la sequenza (unrollando nel ciclo))
    # e poi accoda tutti i risultati in unico vettore lunghissimo. E' lo stesso che encodare un frame alla volta ma cosi fai prima
    # 
    # Ogni cella prende (stato_cella_preced,z, azione,reset_flag) e da in output lo (stato_cella, i dati da dare alla MDN per produrre i successivi 500 frame(500,64,15)
    def unroll(self):
        with tf.variable_scope("lstm_unroll"):
            # split input for time_step: create a "self.seq_length" long list of  (self.batch_size, latent_size+act+rew+rest) tensor from input_seq 
            #len(inputs.shape(self.batch_size, latent_size+act+rest)) = self.seq_length
            inputs = tf.unstack(self.input_seq, axis=1)

            state = self.initial_state 

            zero_c, zero_h = self.zero_state
            outputs = []
            prev = None

            for i in range(self.seq_length):
                # for current i-th time stamp check for each entry of batch if restart_flag is setted to 1  
                # boolean vector of length "batch_size". Indicates with true if that sequence is intial sequences
                restart_flag = tf.greater(self.input_res_flag[:, i], 0.5)
                
                c, h = state

                # where(condition,vector_x,vector_y): assign y[i] to x[i] where condition[i] is true
                # if restart is 1, then reset lstm state to zero
                c = tf.where(restart_flag, zero_c, c)
                h = tf.where(restart_flag, zero_h, h)

                inp = inputs[i]
                    
                # feed current input and the state of previous cell (setted to 0 if the episode is resetted) 
                # state is based on C and H (each one with shape of (batch_size,lstm_state)) while OUTPUT is a single element with shape (batch_size,lstm_state) 
                output, state = self.cell(inp, tf.nn.rnn_cell.LSTMStateTuple(c, h))

                outputs.append(output)
            

            self.next_state = state
            #self.outputs: a list long "seq_len" of lstm states (each one of shape=(batch_size, lstm_state)) 
            self.outputs = outputs  
            



    def init_cell(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(self.dim_cell_state) #init cell with 512 hidden state size
        
        # the state of LSTM is based on two element. C and H each other zero-filled tensors with shape of (batch_size, dim_cell_state) 
        # LSTMStateTuple(c=<tf.Tensor 'LSTMCellZeroState/zeros:0' shape=(100, 512) dtype=float32>, h=<tf.Tensor 'LSTMCellZeroState/zeros_1:0' shape=(100, 512) dtype=float32>)
        self.zero_state = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        #set the initial_state to zero for the first rollout
        self.initial_state = self.zero_state



    def define_placeholder(self):
        

        if self.lstm_type == "INFERENCE":
            self.input_obs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length, self.latent_size], name="input_enc_frames")
            self.input_action = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length, self.action_space], name="input_action")
            self.input_rewards = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length], name="input_rewards")
            self.input_res_flag = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length], name="input_restart") 
        else:
            #the batch data must be full, (remove the -1)
            #shape = [None, None, self.latent_size] -> [batch_size, seq_length,self.latent_size] 
            #the initial batch must be 1 step more length to store also the targets (data from 0 to N, target from 1 to N+1)
            self.batch_obs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length+1, self.latent_size], name="batch_enc_frames")
            self.batch_action = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length+1, self.action_space], name="batch_action")
            self.batch_rewards = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length+1], name="batch_rewards")
            self.batch_restart_flags = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_length+1], name="batch_restart")

            self.input_obs = self.batch_obs[:, 0:-1, :]
            self.input_action = self.batch_action[:, 0:-1, :]
            self.input_res_flag = self.batch_restart_flags[:, 0:-1]
            self.input_rewards = self.batch_rewards[:, 0:-1]

            self.target_obs = self.batch_obs[:, 1:, :]
            self.target_restart = self.batch_restart_flags[:, 1:]
            self.target_rewards = self.batch_rewards[:, 1:]

            # to implement the learning rate decay
            self.curr_learn_rate = tf.Variable(self.learning_rate, trainable=False)
            global_step = tf.Variable(0, name='global_step', trainable=False)
        
            # calculate new learning rate value
            self.inc_global_step = tf.assign(global_step, global_step+1)

        
        # Concatenates input_obs,input_action, input_rewards, input_restart in one single vector (from obs to obs+act+rew+res)
        # self.input_seq = shape=(self.batch_size, self.seq_length, latent_size+act+rew+rest)
        self.input_seq = tf.concat([    self.input_obs,
                                        tf.reshape(self.input_action, [self.input_obs.shape[0], self.seq_length, self.action_space]),
                                        tf.reshape(self.input_rewards, [self.input_obs.shape[0], self.seq_length, 1]),    
                                        tf.reshape(self.input_res_flag, [self.input_obs.shape[0], self.seq_length, 1])
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


    # Collect assignment op to variable for restoring weights from file
    def collect_assign_ops(self):
        #list of all trainable variables
        t_vars = tf.trainable_variables()

        for var in t_vars:
            # the shape of the variable
            pshape = var.get_shape() 

            # use pshape to set placeholder shape
            pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')

            # store in assign_op the assignment of placeholder to variable (next I will use this operation by feeding variable weights into placeholder)
            assign_op = var.assign(pl) 

            #map all assignments to the relative var
            self.assign_ops[var] = (assign_op, pl) 

            print(var)
        
    #######################################
    # Methods to store model
    #######################################
    
    def get_model_params(self):
        with self.graph.as_default():    
            model_params = []
            t_vars = tf.trainable_variables()
            for var in t_vars:
                p = self.sess.run(var) #get weights of this variable
                params = np.round(p*10000).astype(np.int).tolist()
                model_params.append(params)
            return model_params


    def save_json(self, jsonfile='models/lstm.json'):        
        model_params = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
        print("Model saved!")
        
    #def load_json(self, jsonfile='models/lstm.json'):
    def load_json(self, jsonfile='models/lstm.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
        print("Model loaded!")
    
    def set_model_params(self, params):
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                #get the variable shape 
                pshape = tuple(var.get_shape().as_list()) 
                
                # get the parameter values
                p = np.array(params[idx]) 
                assert pshape == p.shape, "inconsistent shape"

                #get assignment operation between variable and placeholder from dictionary
                assign_op, pl = self.assign_ops[var] 

                #feed loaded value into placeholder and assign it to the variable
                self.sess.run(assign_op, feed_dict={pl.name: p/10000.})  
                
                idx += 1