# MODEL DETAILS ---------------------------------------------------------------------

# Training name
training_name = 'SlimConv2D_SQ_120e_2bit'

# Initial thresholds from transformer
initial_thresholds = [322,710,1699]

# Gaussian noise parameters
NOISE_MU = 0.0
NOISE_SIGMA = 120.0 # e-

# Precision of input data
N_BITS = 2

# Number of trainings to run
import argparse
parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('-n','--num', help='Number of trainings to run', required=True)
args = vars(parser.parse_args())

n_tries = int(args['num'])
print("Running " + str(n_tries) + " trainings")

# IMPORTS ---------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import Sequence
from keras.layers import Conv2D, MaxPooling2D
from qkeras import *

from keras.utils import Sequence
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, Callback
import csv

from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator
from loss import custom_sse_loss
from SoftQuantizeLayer import SoftQuantizeLayer
from AnnealingScheduler import AnnealingScheduler

from models import *

pi = 3.14159265359
maxval=1e9
minval=1e-9

# TRAINING DATA ---------------------------------------------------------------------

dataset_base_dir = "/uscms/home/bweiss/nobackup/smart-pixels/"
tfrecords_base_dir = "/uscms/home/jennetd/nobackup/smart-pixels/tfrecords/"

dataset_dir_train = os.path.join(dataset_base_dir, "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets", 'train_contained/')
dataset_dir_val = os.path.join(dataset_base_dir, "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets", 'test_contained/')

tfrecords_dir_train = os.path.join(tfrecords_base_dir, "TFR_train",'3sr_16x16_'+str(int(NOISE_SIGMA))+'eN_raw_slim')
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val",'3sr_16x16_'+str(int(NOISE_SIGMA))+'eN_raw_slim')

seeds = []

# BEGIN LOOP ---------------------------------------------------------------------
for i in range(n_tries):

    print("Running training " + str(i) + " of model " + training_name)   
    with open('log_'+training_name+'.txt','a') as f:
        f.write("Running training " + str(i) + " of model " + training_name + "\n")
    
    seed = random.randint(0, 1000)
    while True:
        if seed in seeds:
            seed = random.randint(0, 1000)
        else:
            break

    print("Seed: ", seed)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Seed: ' + str(seed) + "\n")
    
    # Loading pre-generated TFRecords
    validation_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= tfrecords_dir_val,
        shuffle=True,
        seed=seed,
        quantize=False,
    )

    training_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir = tfrecords_dir_train,
        shuffle=True,
        seed=seed,
        quantize=False,
    )
    
    print("Initial thresholds: ", initial_thresholds)
    with open('log_'+training_name+'.txt','a') as f:
        f.write("Initial thresholds: " + str(initial_thresholds) + "\n")

    model = CreateSQModel_Conv2DSlim(shape = (16,16,2), 
                          n_filters=5,
                          pool_size=3,
                          initial_thresholds=initial_thresholds)

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3, clipnorm=1.0),
        loss=custom_sse_loss,
    )

    fingerprint = '%08x' % random.randrange(16**8)

    print("Fingerprint: ", fingerprint)
    with open('log_'+training_name+'.txt','a') as f:
        f.write('Fingerprint: ' + str(fingerprint) + "\n")
    
    base_dir = f'/uscms/home/jennetd/nobackup/smart-pixels/noise-paper/trained_models/model-{fingerprint}-{training_name}-checkpoints'
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    checkpoint_filepath = os.path.join(checkpoints_dir, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5')
    
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True) 
    
    early_stopping_patience = 500
    es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

    mcp = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_freq='epoch'
    )

    csv_logger = CSVLogger(f'{base_dir}/training_log.csv', append=True)
    scheduler_callback = AnnealingScheduler(
        schedule='cosine',  
        target_layer_name='soft_quantizer_output', 
        initial_k=1.0,
        final_k=67.0, 
        verbose=1     
    )

    quantizer_logger = SoftQuantizeLoggerCallback(
        log_filepath=f"{base_dir}/soft_quantizer_state_log.csv", # New, more descriptive filename
        layer_name="soft_quantizer_output"
    )

    history = model.fit(
        x=training_generator,
        validation_data=validation_generator,
        callbacks=[es,mcp, csv_logger, scheduler_callback, quantizer_logger],
        epochs=5000,
        shuffle=True,
        verbose=1
    )

    del validation_generator
    del training_generator
    del model
    del history

# END LOOP ---------------------------------------------------------------------

print("Finished!")
