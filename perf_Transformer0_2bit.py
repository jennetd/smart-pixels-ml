# Training name
training_name = 'Transformer0e_2bit'

# Number of trainings to run
n_tries = 2

# Gaussian noise parameters
NOISE_MU = 0.0
NOISE_SIGMA = 0.0 # e-

# Precision of input data
N_BITS = 2

# IMPORTS ---------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os, subprocess
import random

from keras.utils import Sequence
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from models import *
from plotting import *

# DATA ---------------------------------------------------------------------

dataset_base_dir = "/uscms/home/bweiss/nobackup/smart-pixels/"
tfrecords_base_dir = "/uscms/home/jennetd/nobackup/smart-pixels/tfrecords"

dataset_dir_val = os.path.join(dataset_base_dir, "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets", 'test_contained/')
tfrecords_dir_val   = os.path.join(tfrecords_base_dir, "TFR_val",'3sr_16x16_'+str(int(NOISE_SIGMA))+'eNoise_test')

batch_size = 5000
val_batch_size = 5000
val_file_size = len(os.listdir(dataset_dir_val))

def grep(string, filename):

    out_list = []

    cmd = "grep '" + string + "' " + filename
    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Equivalent of grep
    #grep_process = subprocess.Popen(['grep', string, filename], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait till process to exit
    #out, error = grep_process.communicate()

    # Get output data
    cmd_out = result.stdout.splitlines()
    for line in cmd_out:
        out_list += [line.split(':')[1].strip()]

    return out_list

seeds = grep('Seed','log_Transformer0_2bit*')
print("Seeds = ", seeds)

fingerprints = grep('Fingerprint','log_Transformer0_2bit*')
print("Fingerprints = ", fingerprints)
    
residuals = {}

# BEGIN LOOP ---------------------------------------------------------------------
for i in range(len(fingerprints)):

    base_dir = './trained_models/model-'+fingerprints[i]+'-'+training_name+'-checkpoints/'
    checkpoint_dir = base_dir + 'checkpoints/'
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.hdf5')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)

    model = create_vit_model(input_shape=(16,16,2),   
                             patch_size=(3,4),        
                             embed_dim=64,           
                             num_heads=4,            
                             ff_dim=128,              
                             num_layers=4,            
                             dropout=0.1,        
                             n_bits=N_BITS,
                             final_outputs=14         
                            )

    print(f"Loading model from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

    test_generator = OptimizedDataGenerator(
        load_from_tfrecords_dir= tfrecords_dir_val,
        shuffle=True,
        seed=int(seeds[i]),
        quantize=False,
    )

    sf = test_generator.labels_scale

    scale_factor = {}
    scale_factor['x'] = sf[0]
    scale_factor['y'] = sf[1]
    scale_factor['cotA'] = sf[2]
    scale_factor['cotB'] = sf[3]

    df = construct_df(model, test_generator, scale_factor)
    df.to_parquet(base_dir+"residuals.parquet")
    residuals[i] = df

    """
    fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=(8,6))
    pull_plot(df, axes[0][0],'pullx',r'$x$ pull')
    pull_plot(df, axes[0][1],'pully',r'$y$ pull')
    pull_plot(df, axes[1][0],'pullcotA',r'$\cot\alpha$ pull')
    pull_plot(df, axes[1][1],'pullcotB',r'$\cot\beta$ pull')

    #save_fig_path = os.path.join(base_dir, 'Pull.png')
    #plt.savefig(save_fig_path)

    plt.show()

    fig, axes = plt.subplots(2,2,figsize=(8,6))
    fig.tight_layout(pad=4.0)
    residual_plot(axes[0][0],df,'xtrue','x',r'$x$ [um]')
    axes[0][0].plot([-25,-25],[-10,10],color='gray',linestyle=':')
    axes[0][0].plot([25,25],[-10,10],color='gray',linestyle=':')
    residual_plot(axes[0][1],df,'ytrue','y',r'$y$ [um]')
    axes[0][1].plot([-6.25,-6.25],[-2,2],color='gray',linestyle=':')
    axes[0][1].plot([6.25,6.25],[-2,2],color='gray',linestyle=':')
    residual_plot_deg(axes[1][0],df,'cotAtrue','cotA',r'$\alpha$ [deg]')
    axes[1][0].plot([90,90],[-10,10],color='gray',linestyle=':')
    residual_plot_deg(axes[1][1],df,'cotBtrue','cotB',r'$\beta$ [deg]')
    axes[1][1].plot([90,90],[-10,10],color='gray',linestyle=':')

    #save_fig_path = os.path.join(base_dir, 'summary.png')
    #plt.savefig(save_fig_path)

    history(base_dir)

    threshold_evolution(base_dir)
    """

# OVERLAY RESIDUALS OF ALL TRAININGS --------------------------------------

# Pixel pitch (um)
xpitch = 50
ypitch = 12.5
# Thickness (um)
D = 100
# Size of pixel array
lenx = 16
leny = 16

fig, ax = plt.subplots(2,4,figsize=(10,4),sharey='row', constrained_layout=True, gridspec_kw={'height_ratios': [1, 2]})

ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[0,3].axis('off')

for i,df in residuals.items():

    print(i)

    # x
    ax[1,0].set_xlabel(r'$R_x$ [um]')
    ax[1,0].hist(df["residualx"],histtype='step',bins=np.linspace(-3*xpitch,3*xpitch,30),color=colors[i])
    ax[1,0].set_yscale('log')

    # y
    ax[1,1].set_xlabel(r'$R_y$ [um]')
    ax[1,1].hist(df["residualy"],histtype='step',bins=np.linspace(-3*ypitch,3*ypitch,30),color=colors[i])

    # alpha
    ax[1,2].set_xlabel(r'$R_\alpha$ [deg]')
    ax[1,2].hist(df["residualA"],histtype='step',bins=np.linspace(-60,60,30),color=colors[i])
    
    # beta
    ax[1,3].set_xlabel(r'$R_\beta$ [deg]')
    ax[1,3].hist(df["residualB"],histtype='step',bins=np.linspace(-60,60,30),color=colors[i])

plt.tight_layout()
plt.savefig('plots/'+training_name+'_residuals.png')