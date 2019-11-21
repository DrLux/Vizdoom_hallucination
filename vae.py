import numpy as np
import matplotlib.pyplot as plt 
import parameters #my file

# for tf1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import json

class VAE(object):
    
    def __init__(self,dataset):
        self.latent_size = parameters.LATENT_SIZE
        self.learning_rate = parameters.VAE_LEARNING_RATE
        self.train_epochs = parameters.TRAIN_EPOCHS
        self.frame_shape = parameters.FRAME_SHAPE
        self.dataset = dataset
        self.assign_ops = {} #dictionary to collect assignment operations to reload the model weights
        
        self.input_batch = None
        self.latent_vec = []
        self.output_batch = None
        self.loss = 0.0
        self.optimizer = None


        self.build_computational_graph()
        

    def build_computational_graph(self):
        self.input_batch = tf.placeholder(tf.float32, shape=[None, self.frame_shape[0], self.frame_shape[1], 3])
        self.encoder() 
        self.reparametrization_trick()
        self.decoder()
        self.define_optimizer()


        #Init Session
        init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init_vars)
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
            img_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.input_batch - self.output_batch),reduction_indices = [1,2,3]))
                
            # kl loss for two gaussian
            # formula: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
            #kl_loss = - 0.5 * tf.reduce_sum((1 + tf.log(tf.square(latent_std_dev)) - tf.square(latent_mean) - tf.square(latent_std_dev)), reduction_indices = 1)

            ###kl-loss from hardmaru "world_model" repo
            kl_loss = - 0.5 * tf.reduce_sum((1 + self.latent_std_dev - tf.square(self.latent_mean) - tf.exp(self.latent_std_dev)),reduction_indices = 1)          
            kl_loss = tf.reduce_mean(kl_loss)

            self.loss = img_loss + kl_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def train_vae(self, checkpoint = False):
        for epoch in range(self.train_epochs):
            print("\n Epoch: ", epoch)
            batches = self.dataset.get_frames_batches()
            for b in batches:
                loss_value,_ = self.sess.run([self.loss,self.optimizer], {self.input_batch: b})
                print("loss_value: ", loss_value)
            if checkpoint:
                self.save_json()

    def post_process_frame(self,reconstructed_img):
        reconstructed_img = np.round(reconstructed_img * 255.).astype(np.uint8)
        img = reconstructed_img.reshape(64, 64, 3) #remove batch dimension
        return img
    
    #img could be a single image or a batch of images
    def encode_input(self,img):
        norm_input_frame = np.float32(img.astype(np.float)/255.0)        
        
        #if img is a single image, add a batch dimension
        if img.shape == 3:
            input_batch = norm_input_frame.reshape(1, 64, 64, 3)
        else:
            input_batch = norm_input_frame
        latent_vec = self.sess.run(self.latent_vec, {self.input_batch: input_batch})
        return latent_vec

    def decode_latent_vec(self,latent_v):
        reconstructed_frames = self.sess.run(self.output_batch, {self.latent_vec: latent_v})
        return reconstructed_frames

    def synthesize_image(self,img):
        latent_vec = self.encode_input(img)
        reconstructed_img = self.decode_latent_vec(latent_vec)
        return self.post_process_frame(reconstructed_img)  

    # encode dataset of images e return a dataset of latent vectors
    # this routine split dataset, encode each single parts and concatenate them togheter 
    def encode_dataset(self):
        dataset_to_encode = self.dataset.frame_dataset
        split_dimension = 1000
        split_counter = int(len(dataset_to_encode) / split_dimension)
        encoded_dataset = []
        for i in range(split_counter):
            print("Encoded ", i*split_dimension, " frames!")
            splitted_part = dataset_to_encode[i * split_dimension:(i+1) * split_dimension]
            encoded_dataset.append(self.encode_input(splitted_part))
        encoded_dataset = np.concatenate(encoded_dataset) #from list to single np.array
        self.dataset.encoded_frame_dataset = encoded_dataset


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


    def save_json(self, jsonfile='models/vae.json'):
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