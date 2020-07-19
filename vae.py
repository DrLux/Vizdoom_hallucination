import numpy as np
import matplotlib.pyplot as plt 
import parameters #my file
import os
from PIL import Image


# for tf1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class VAE(object):
    
    def __init__(self,dataset):
        self.initial_learning_rate = parameters.VAE_LEARNING_RATE
        self.latent_size = parameters.LATENT_SIZE
        self.train_epochs = parameters.TRAIN_EPOCHS
        self.frame_shape = parameters.FRAME_SHAPE
        self.dataset = dataset
        self.assign_ops = {} #dictionary to collect assignment operations to reload the model weights
        
        self.curr_learn_rate = None 
        self.input_batch = None
        self.latent_vec = []
        self.output_batch = None
        self.loss = 0.0
        self.optimizer = None
   
        self.build_computational_graph()
        

    def build_computational_graph(self):
        self.input_batch = tf.placeholder(tf.float32, shape=[None, self.frame_shape[0], self.frame_shape[1], 3])
            
        # to implement the learning rate decay
        self.curr_learn_rate = tf.Variable(parameters.VAE_LEARNING_RATE, trainable=False)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # calculate new learning rate value
        self.inc_global_step = tf.assign(global_step, global_step+1)
        self.reset_global_step =  tf.assign(global_step, 0)
        
        self.encoder() 
        self.reparametrization_trick()
        self.decoder()
        self.define_optimizer()


        #Init Session
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init_vars)
        tf.set_random_seed(1)
        self.collect_assign_ops()

    # data shape:  (batch_size, self.train_epochs, frame_shape, 3)
    def encoder(self):
        with tf.variable_scope("Encoder"):
            input_layer = tf.layers.conv2d(inputs=self.input_batch, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv1", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv1/Relu:0", shape=(batch_size, 31, 31, 32), dtype=float32)

            hidden_layer = tf.layers.conv2d(input_layer, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv2", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv2/Relu:0", shape=(batch_size, 14, 14, 64), dtype=float32)

            hidden_layer = tf.layers.conv2d(hidden_layer, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv3", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv3/Relu:0", shape=(batch_size, 6, 6, 128), dtype=float32)

            hidden_layer = tf.layers.conv2d(hidden_layer, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, name="enc_conv4", reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/enc_conv4/Relu:0", shape=(1batch_size, 2, 2, 256), dtype=float32)

            last_layer = tf.layers.flatten(hidden_layer) # same effect of tf.reshape(hidden_layer, [-1, 2*2*256])        
            #Tensor("Encoder/flatten/Reshape:0", shape=(batch_size, 1024), dtype=float32)
            
            # Latent Vector: mean and st_dev
            self.latent_mean = tf.layers.dense(last_layer, units=self.latent_size, name='mean', reuse=tf.AUTO_REUSE)
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)
            
            # std_dev mus be always > 0
            self.latent_std_dev = tf.exp(tf.layers.dense(last_layer, units=self.latent_size, reuse=tf.AUTO_REUSE)/ 2.0)
            #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)


    def reparametrization_trick(self):
        # allocate a vector of random epsilon for rep.trick
        epsilon = tf.random_normal(tf.shape(self.latent_std_dev), name='epsilon')
        #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)

        # Sample a vector z by mean + (str_dev * epsilon)
        self.latent_vec = self.latent_mean + tf.multiply(epsilon, self.latent_std_dev)
        #Tensor("Encoder/add:0", shape=(batch_size, 8), dtype=float32)


    def decoder(self):
        with tf.variable_scope("Decoder"):
            input_layer = tf.layers.dense(self.latent_vec, 4*256, name="dec_input_fullycon",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_input_fullycon/BiasAdd:0", shape=(10, 1024), dtype=float32)
            
            input_layer = tf.reshape(input_layer, [-1, 1, 1, 4*256])
            #Tensor("Decoder/Reshape:0", shape=(10, 1, 1, 1024), dtype=float32)

            hidden_layer = tf.layers.conv2d_transpose(input_layer, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv1/Relu:0", shape=(10, 5, 5, 128), dtype=float32)    

            hidden_layer = tf.layers.conv2d_transpose(hidden_layer, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv2/Relu:0", shape=(10, 13, 13, 64), dtype=float32)

            hidden_layer = tf.layers.conv2d_transpose(hidden_layer, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3",reuse=tf.AUTO_REUSE)
            #Tensor("Decoder/dec_deconv3/Relu:0", shape=(10, 30, 30, 32), dtype=float32)

            # NB: we use a sigmoid function so the output values will be between 0 and 1 
            self.output_batch = tf.layers.conv2d_transpose(hidden_layer, 3, 6, strides=2, activation=tf.nn.sigmoid, name="reconstructor",reuse=tf.AUTO_REUSE)  
            #output_batch:  Tensor("Decoder/reconstructor/Sigmoid:0", shape=(batch_size, 64, 64, 3), dtype=float32)

   
    def define_optimizer(self):
        with tf.name_scope('loss'):
        
            # reconstruction loss (MEAN SQUARE ERROR between 2 images):  
            self.img_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_batch - self.output_batch),reduction_indices = [1,2,3]))
            
                
            # kl loss for two gaussian
            # formula: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
            #kl_loss = - 0.5 * tf.reduce_sum((1 + tf.log(tf.square(latent_std_dev)) - tf.square(latent_mean) - tf.square(latent_std_dev)), reduction_indices = 1)

            ###kl-loss from hardmaru "world_model" repo
            self.kl_loss = - 0.5 * tf.reduce_sum((1 + self.latent_std_dev - tf.square(self.latent_mean) - tf.exp(self.latent_std_dev)),reduction_indices = 1)          
            self.kl_loss = tf.reduce_mean(self.kl_loss)

            self.loss = self.img_loss + self.kl_loss
            self.optimizer = tf.train.AdamOptimizer(self.curr_learn_rate).minimize(self.loss)
        
  
    # I implemented the learning rate decay but i do not use it
    # I manually decrease the leargnin rate witch switch case because it works better
    def train_vae(self, checkpoint = False):
        batches = self.dataset.split_set_for_vae("train_set")
        log_loss_value = []
        #log_learn_rate = []
        global_step = 0
        curr_learn_rate = self.initial_learning_rate
        record_loss = 10000
        
        for epoch in range(self.train_epochs):
            print("\n\t ****** Epoch: ******", epoch)
            np.random.shuffle(batches)
            
            '''if (epoch == 20):
                curr_learn_rate = 0.0008
            elif (epoch == 45):
                curr_learn_rate = 0.0003
            elif (epoch == 50):
                curr_learn_rate = 0.0001'''
            
            for b in batches:
                loss_value,_,img_loss,kl_loss = self.sess.run([self.loss,self.optimizer,self.img_loss,self.kl_loss], {self.input_batch: b,self.curr_learn_rate: curr_learn_rate})

                print("\n")
                print("global_step: ", global_step)
                print("curr_learn_rate: ", curr_learn_rate)
                print("loss_value: ", loss_value)
                print("epoch: ", epoch)
                #print("img_loss: ",img_loss)
                #print("kl_loss:", kl_loss)
                print("\n")

                log_loss_value.append(loss_value)
                #log_learn_rate.append(curr_learn_rate)
                
                # Only when achieve a better result save checkpoint
                if loss_value < record_loss:
                    record_loss = loss_value

                    # Is pointless saving so frequently as with the first epochs
                    if epoch >= 9:

                        if epoch%2 == 0:
                            self.save_json()
                        else:
                            self.save_json()
                            self.save_json(jsonfile='models/backup_vae_model.json')

                        f = open("models/vae_best_loss_value.txt", "a")
                        f.write("\n"+str(epoch)+" -> "+str(record_loss))
                        f.close()


                        


                # Calculate new learning rate
                global_step = self.sess.run(self.inc_global_step)
                #curr_learn_rate = initial_learning_rate * (parameters.VAE_LEARNING_RATE_DECAY ** global_step) + parameters.VAE_MIN_LEARNING_RATE

                
        print("Final loss: ", record_loss)
        return log_loss_value



    def test_vae(self): 
        self.load_json()
        curr_learn_rate = parameters.VAE_MIN_LEARNING_RATE
        log_loss_value = []
        print("Testing with learning rate: ", curr_learn_rate)
        batches = self.dataset.split_set_for_vae("test_set")

        for b in batches:
            loss_value,_ = self.sess.run([self.loss,self.optimizer], {self.input_batch: b,self.curr_learn_rate: curr_learn_rate})
            log_loss_value.append(loss_value)

        print(np.array(log_loss_value).mean())

        return log_loss_value


    def post_process_frame(self,reconstructed_img):
        reconstructed_img = np.round(reconstructed_img * 255.).astype(np.uint8)
        img = reconstructed_img.reshape(64, 64, 3) #remove batch dimension
        #reconstructed_frame = 255-reconstructed_frame #module 255 to restore original graphic effect
        return img
    
    #img could be a single image or a batch of images
    def encode_input(self,img):
        norm_input_frame = np.float32(img.astype(np.float)/255.0)        
        
        #if img is a single image, add a batch dimension
        if len(img.shape) == 3:
            input_batch = norm_input_frame.reshape(1, 64, 64, 3)
        else:
            input_batch = norm_input_frame
        latent_vec = self.sess.run(self.latent_vec, {self.input_batch: input_batch})
        return latent_vec

    def decode_latent_vec(self,latent_v):
        if len(latent_v.shape) == 1:
            latent_v = latent_v.reshape(1, 64)
        reconstructed_frames = self.sess.run(self.output_batch, {self.latent_vec: latent_v})
        return reconstructed_frames

    def synthesize_image(self,img):
        latent_vec = self.encode_input(img)
        reconstructed_img = self.decode_latent_vec(latent_vec)
        return self.post_process_frame(reconstructed_img)  

    # encode dataset of images e return a dataset of latent vectors
    # this routine split dataset, encode each single parts and concatenate them togheter 
    def encode_dataset(self):
        self.dataset.load_only_frame()
        dataset_to_encode = self.dataset.dataset["obs"] 
        encoded_dataset = []

        for batch in dataset_to_encode:
            encoded_dataset.append(self.encode_input(batch))
        
        encoded_dataset = np.stack(encoded_dataset)

        self.dataset.dataset["enc_obs"] = encoded_dataset
        self.dataset.store_single_dataset("dataset/enc_obs.npz",encoded_dataset)    


    ###############################################
    # Routine to dump and restore model (below) is based on: 
    # https://github.com/hardmaru/WorldModelsExperiments/blob/244f79c2aaddd6ef994d155cd36b34b6d907dcfe/doomrnn/doomrnn.py#L76
    ###############################################

    def get_model_params(self):
        model_params = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            p = self.sess.run(var) #get weights of this variable
            params = np.round(p*10000).astype(np.int).tolist()
            model_params.append(params)
        return model_params


    def save_json(self, jsonfile='models/vae.json' ):
        if not os.path.exists("models"):
            os.makedirs("models")

        model_params = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
        print("Model saved!")

    def load_json(self, jsonfile='models/vae.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
        print("Model loaded!")

    
    
    def set_model_params(self, params):
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