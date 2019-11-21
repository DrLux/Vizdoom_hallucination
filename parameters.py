#VAE hyperparameters
VAE_LEARNING_RATE = 0.00005
VAE_BATCH_SIZE = 100
TRAIN_EPOCHS = 3 #10 default
FRAME_SHAPE = [64,64]
LATENT_SIZE = 64

#LSTM hyperparameters
MIXTURE = 5 #self.num_mixture -> 5 mixtures
LATENT_SIZE = 64 #hps.seq_width -> 64 channels
INPUT_SIZE = LATENT_SIZE #size of input (latent vector )
DIM_CELL_STATE = 512 #cell state
LSTM_LEARNING_RATE = 0.01
TINY = 1e-6 # to avoid NaNs in logs
RECURRENT_DROPOUT_PROB = 0.90
LSTM_BATCH_SIZE = 100
SEQ_LENGTH = 500 #100 timesteps

#Dataset hyperparameters
DATASET_SIZE =  500000
LSTM_NUM_BATCH = int(DATASET_SIZE/(LSTM_BATCH_SIZE * SEQ_LENGTH))
VAE_NUM_BATCHES = int(DATASET_SIZE/VAE_BATCH_SIZE)