import os, csv

import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from DG.OptimizedDataGenerator_v2p5 import OptimizedDataGenerator
from loss import custom_loss
from SoftQuantizeLayer import SoftQuantizeLayer
from AnnealingScheduler import AnnealingScheduler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, Callback

def hard_quantize(x, levels, thresholds):
        x_reshaped = tf.expand_dims(x, axis=-1) 
        is_grt_th = x_reshaped > thresholds 
        indices = tf.reduce_sum(tf.cast(is_grt_th, dtype=tf.int32), axis=-1) 
        return tf.cast(tf.gather(levels, indices),tf.float32)

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv2d_network(var, n_filters=5, kernel_size=3):
    var = QSeparableConv2D(
        n_filters,kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def conv1d_network(var, n_filters=5, kernel_size=3):
    nrows = var.shape[1] # 13, for now
    ncols = var.shape[2] # 20, for now
    timeslices = var.shape[3] # either 20 or 2, for now
    proj_x = AveragePooling2D(
        pool_size=(1, 16),
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_x"
    )(var)
    proj_x = Reshape((nrows, timeslices), name="reshape_proj_x")(proj_x)
    proj_y = AveragePooling2D(
        pool_size=(nrows, 1),
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_y"
    )(var)
    proj_y = Reshape((ncols, timeslices), name="reshape_proj_y")(proj_y)

    proj_x = QConv1D(
        n_filters,kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_x"
    )(proj_x)

    proj_y = QConv1D(
        n_filters,kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_y"
    )(proj_y)

    var = Concatenate(axis=1, name="concatenate")([proj_x, proj_y])
    var = QActivation("quantized_tanh(4, 0, 1)", name="activation_tanh_1")(var)

    return var

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten()(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten()(proj_y)

    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = QActivation("quantized_relu(bits=13, integer=5)")(proj_x)

    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = QActivation("quantized_relu(bits=13, integer=5)")(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return var


# Max Conv2D model with hard quantization
def CreateHQModel(shape, n_filters, output, pool_size, conv_kernel_size=3, levels=[0,1,2,3], thresholds=[0,100,200,300]):
    x_base = x_in = Input(shape)

    # Hard quantize
    stack = hard_quantize(x_base, levels, thresholds)  
    stack = conv2d_network(stack, n_filters=n_filters, kernel_size=conv_kernel_size) 
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

# Max Conv2D model with soft quantization
def CreateSQModel(shape, output, n_filters, pool_size, conv_kernel_size=3, levels=[0,1,2,3], initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = conv2d_network(x_base, n_filters=n_filters, kernel_size=conv_kernel_size)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=output)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQModel_Conv2DSlim(shape, n_filters, pool_size, levels=[0,1,2,3], initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = conv2_network(x_base,n_filters=n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQModel_Conv1DSlim(shape, n_filters, levels=[0,1,2,3], initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = conv1d_network(x_base, n_filters=n_filters)
    stack = var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def CreateSQModel_MLPSlim(shape, levels=[0,1,2,3], initial_thresholds=[10,100,200]):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds = initial_thresholds,
        initial_levels = levels,
        threshold_offset = 0,
        trainable_levels=False,
        trainable_thresholds=True,
        initial_k=1.0,
        trainable_k=False,
        name='soft_quantizer_output'
    )(x_base)
    stack = mlp_encoder_network(x_base)
    stack = var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

# Trasformer model

class PatchExtractor(layers.Layer):
    """Extract 2D patches from images."""
    def __init__(self, patch_size=(3,7)):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # images: (batch, H, W, C)
        patch_h, patch_w = self.patch_size
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, patch_h, patch_w, 1),
            rates=(1,1,1,1),
            padding='VALID')
        # Now `patches` has shape: 
        #   (batch, H//patch_h, W//patch_w, patch_h*patch_w*C)
        # Flatten the 2D grid of patches => (batch, num_patches, patch_dim)
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(
            patches,
            [batch_size, -1, patch_dims]
        )
        return patches

class PatchEncoder(layers.Layer):
    """Linear embedding + learnable positional encoding."""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection  = layers.Dense(embed_dim)
        self.pos_embed   = tf.Variable(
            initial_value=tf.zeros((1,num_patches,embed_dim)),
            trainable=True,
            name="pos_embedding"
        )

    def call(self, patch_batch):
        # Linear projection of patch tokens
        projected = self.projection(patch_batch)
        # Add learnable positional embeddings
        return projected + self.pos_embed

def transformer_encoder(inputs,
                        head_size,
                        num_heads,
                        ff_dim,
                        dropout=0.1):
    # LayerNorm + Multi-head attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads,
                                  key_dim=head_size,
                                  dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
  
    # Next LN + feed-forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation="linear")(x)
    x = layers.Dropout(dropout)(x)
  
    return x + res

def create_vit_model(input_shape=(16,16,2),
                     patch_size=(3,7),
                     embed_dim=64,
                     num_heads=4,
                     ff_dim=128,
                     num_layers=4,
                     dropout=0.1,
                     threshold_offset=0.0,
                     n_bits = 2,
                     initial_thresholds = [247.8, 668.4, 1662.9],
                     final_outputs=14):
  # Input
  inp = layers.Input(shape=input_shape, name="raw_input")

  # q_out = SoftQuantizeLayer(
  #     n_bits=2,
  #     initial_range=[-1.0, 1.0],
  #     trainable_levels=False,
  #     trainable_thresholds=True,
  #     initial_k=1.0,
  #     trainable_k=True,
  #     name="soft_quantizer_output"
  # )(inp)

  q_out = SoftQuantizeLayer(
      n_bits=n_bits,
     #   initial_range=[-1.0, 1.0], # This shoudl be automatically used for the levels
      initial_levels=np.array([i for i in range(0,np.power(2,n_bits))], dtype=np.float32),
      threshold_offset=threshold_offset,
      initial_thresholds=initial_thresholds,
      trainable_levels=False,
      trainable_thresholds=True,
      initial_k=1.0,
      trainable_k=True,
      name="soft_quantizer_output"
  )(inp)

  # 1) Extract patches
  patches = PatchExtractor(patch_size=patch_size)(q_out)
  
  # Calculate how many patches we extracted:
  #   (H // patch_h) * (W // patch_w)
  # Must do it statically if possible:
  # e.g. 13//3=4, 21//7=3 => 12 patches total
  H, W, C = input_shape
  ph, pw  = patch_size
  num_patches = (H // ph) * (W // pw)

  # 2) Encode patches (linear projection + positional embedding)
  encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

  # 3) Apply multiple Transformer encoder blocks
  x = encoded_patches
  for _ in range(num_layers):
    x = transformer_encoder(x,
                            head_size=embed_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
  
  # 4) Flatten and final Dense
  x = layers.LayerNormalization(epsilon=1e-6)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(final_outputs, activation='linear')(x)

  # Create model
  model = keras.Model(inputs=inp, outputs=outputs)
  return model

def create_fullprecision_model(input_shape=(16,16,2),
                               patch_size=(3,7),
                               embed_dim=64,
                               num_heads=4,
                               ff_dim=128,
                               num_layers=4,
                               dropout=0.1,
                               final_outputs=14):
  # Input
  inp = layers.Input(shape=input_shape, name="raw_input")

  # 1) Extract patches
  patches = PatchExtractor(patch_size=patch_size)(inp)
  
  # Calculate how many patches we extracted:
  #   (H // patch_h) * (W // patch_w)
  # Must do it statically if possible:
  # e.g. 13//3=4, 21//7=3 => 12 patches total
  H, W, C = input_shape
  ph, pw  = patch_size
  num_patches = (H // ph) * (W // pw)

  # 2) Encode patches (linear projection + positional embedding)
  encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

  # 3) Apply multiple Transformer encoder blocks
  x = encoded_patches
  for _ in range(num_layers):
    x = transformer_encoder(x,
                            head_size=embed_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
  
  # 4) Flatten and final Dense
  x = layers.LayerNormalization(epsilon=1e-6)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(final_outputs, activation='linear')(x)

  # Create model
  model = keras.Model(inputs=inp, outputs=outputs)
  return model


def create_mf_fullprecision_model(input_shape=(16,16,2),
                               patch_size=(3,7),
                               embed_dim=64,
                               num_heads=4,
                               ff_dim=128,
                               num_layers=4,
                               dropout=0.1,
                               mean_filter_size = (3, 3),
                               final_outputs=14):
  # Input
  inp = layers.Input(shape=input_shape, name="raw_input")

  # This is the mean filter
  inp = AveragePooling2D(
      pool_size=mean_filter_size, 
      strides = (1, 1), 
      padding = "same",
      name = "mean_filter"
  )(inp)

  # 1) Extract patches
  patches = PatchExtractor(patch_size=patch_size)(inp)
  
  # Calculate how many patches we extracted:
  #   (H // patch_h) * (W // patch_w)
  # Must do it statically if possible:
  # e.g. 13//3=4, 21//7=3 => 12 patches total
  H, W, C = input_shape
  ph, pw  = patch_size
  num_patches = (H // ph) * (W // pw)

  # 2) Encode patches (linear projection + positional embedding)
  encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)

  # 3) Apply multiple Transformer encoder blocks
  x = encoded_patches
  for _ in range(num_layers):
    x = transformer_encoder(x,
                            head_size=embed_dim,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout=dropout)
  
  # 4) Flatten and final Dense
  x = layers.LayerNormalization(epsilon=1e-6)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(final_outputs, activation='linear')(x)

  # Create model
  model = keras.Model(inputs=inp, outputs=outputs)
  return model

class SoftQuantizeLoggerCallback(Callback):
    def __init__(self, log_filepath, layer_name="soft_quantizer_output"):
        super().__init__()
        self.log_filepath = log_filepath
        self.layer_name = layer_name
        self.header_written = False

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        try:
            layer = self.model.get_layer(self.layer_name)
            if not hasattr(layer, 'n_bits'):
                 print(f"\nWarning: Layer '{self.layer_name}' is not a SoftQuantizeLayer. Skipping logging.")
                 return
        except ValueError:
            print(f"\nWarning: Layer '{self.layer_name}' not found in the model. Skipping logging.")
            return

        if not self.header_written:
            num_levels = layer.num_levels
            num_thresholds = num_levels - 1
            
            header = ['epoch', 'k']
            header.extend([f'level_{i}' for i in range(num_levels)])
            header.extend([f'threshold_{i}' for i in range(num_thresholds)])

            header.append('raw_first_level')
            header.extend([f'raw_log_level_delta_{i}' for i in range(num_levels - 1)])
            header.append('raw_first_threshold')
            if hasattr(layer, 'log_threshold_deltas'):
                header.extend([f'raw_log_threshold_delta_{i}' for i in range(num_thresholds - 1)])

            with open(self.log_filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            self.header_written = True

        k_val = layer.k.numpy().item()
        levels = layer.levels.numpy().tolist()
        thresholds = layer.thresholds.numpy().tolist()
        
        # first_level = layer.first_level.numpy().item()
        # log_level_deltas = layer.log_level_deltas.numpy().tolist()
        # first_threshold = layer.first_threshold.numpy().item()
        
        row_data = [epoch, k_val]
        row_data.extend(levels)
        row_data.extend(thresholds)
        # row_data.append(first_level)
        # row_data.extend(log_level_deltas)
        # row_data.append(first_threshold)
        
        if hasattr(layer, 'log_threshold_deltas'):
            log_threshold_deltas = layer.log_threshold_deltas.numpy().tolist()
            row_data.extend(log_threshold_deltas)
        
        with open(self.log_filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)