#VAE hyperparameters
VAE_LEARNING_RATE           = 0.004
VAE_BATCH_SIZE              = 100
TRAIN_EPOCHS                = 50
FRAME_SHAPE                 = [64,64]
LATENT_SIZE                 = 64
VAE_LEARNING_RATE_DECAY     = 0.999
VAE_MIN_LEARNING_RATE       = 0.0005



#LSTM hyperparameters
MIXTURE = 5 #self.num_mixture -> 5 mixtures
LATENT_SIZE = 64 #hps.seq_width -> 64 channels
INPUT_SIZE = LATENT_SIZE #size of input (latent vector )
DIM_CELL_STATE = 512 #cell state
LSTM_BATCH_SIZE = 10
SEQ_LENGTH = 500 

# LSTM Fine tuning
LSTM_EPOCH_TRAIN = 10
LSTM_LEARNING_RATE = 0.00005
LSTM_LEARNING_RATE_DECAY = 0.99999
LSTM_MIN_LEARNING_RATE=0.000001


#Dataset hyperparameters
DATASET_SIZE =  100000

#VAE DATASET
VAL_SET_SIZE = 15000
TEST_SET_SIZE = 15000

#LSTM DATASET
LSTM_TEST_CHUNKS = 2 # 2 * batch_size * chunk_size 
LSTM_VAL_CHUNKS = 2  # so 2 * 10 * 500 = 10000