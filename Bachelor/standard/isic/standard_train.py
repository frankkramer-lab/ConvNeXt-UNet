"""
Training of ISIC Dataset with Cbam and Convnext U-net
"""

# Import some libraries
import os
import time
import numpy as np

# Import miscnn libraries
from miscnn import Data_IO, Preprocessor, Neural_Network, Data_Augmentation
from miscnn.data_loading.interfaces import Image_interface
from miscnn.neural_network.metrics import dice_soft, \
                                          dice_crossentropy, tversky_loss
from miscnn.processing.subfunctions import Resize, Normalization
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction
from miscnn.evaluation import split_validation

from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt

# ----------------------------   Paths   ------------------------------------ # 
# Change these paths to where the corresponding data is stored on your system #
# --------------------------------------------------------------------------- #
# Path to test subset of ISIC2018 dataset
training_path = "jonas/Bachelor/isic_seg/training/data"
# Path to validation subset of ISIC2018 dataset
validation_path = "jonas/Bachelor/isic_seg/validation/data"
# Path to where the model should be saved after training
model_path = "jonas/Bachelor/standard/standard_unet.hdf5"

# Initialize Data IO & Image Interface
interface = Image_interface(classes=2, img_type="rgb", img_format="png")

# Get sample list from validation set
sample_list_val = os.listdir(validation_path)

# Initialize Data_IO
data_io = Data_IO(interface, training_path, delete_batchDir=True)
sample_list = [index for index in data_io.get_indiceslist() if index not in sample_list_val]

# Create a custom subfunction to change 255 values (in binary segmentation mask) to 1
class ChangeValues(Abstract_Subfunction):
   #---------------------------------------------#
   #                Initialization               #
   #---------------------------------------------#
  def __init__(self,):
        pass
    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
  def preprocessing(self, sample, training=True):
    seg_temp = sample.seg_data
    if seg_temp is not None:
        sample.seg_data = np.where(seg_temp == 255, 1, seg_temp)
    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
  def postprocessing(self, sample, prediction, activation_output=False):
    pred = prediction
    if prediction is not None:
        prediction = np.where(pred == 1, 255, pred)
    return prediction

# Create a resizing Subfunction to shape 512x512
sf_resize = Resize((512, 512))
sf_zscore = Normalization(mode="z-score")
sf_change = ChangeValues()

# Assemble Subfunction classes into a list
sf = [sf_zscore, sf_resize, sf_change]

# Initialize Data augmentation
aug = Data_Augmentation(cycles=2, scaling=True, rotations=True,
                        elastic_deform=True, mirror=True,
                        gaussian_noise=True)

# Initialize Preprocessor
pp = Preprocessor(data_io, batch_size=6, subfunctions=sf, data_aug=aug,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage")

 # Create the Neural Network models
standard_model = Neural_Network(preprocessor=pp, loss=tversky_loss,
                       metrics=[tversky_loss, dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=5, learning_rate=0.001)

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5,
                          verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
cb_tb = TensorBoard(log_dir="tensorboard", histogram_freq=0, write_graph=True, 
                    write_images=True)
cb_es = EarlyStopping(monitor="val_loss", patience=5)
cb_mc_standard = ModelCheckpoint(filepath=model_path, save_best_only=True)

# Train model and record time
start = time.time()
history = standard_model.evaluate(training_samples=sample_list, validation_samples=sample_list_val, epochs=1000, callbacks=[cb_lr, cb_tb, cb_es, cb_mc_standard])
duration = round(time.time() - start)

# Save training duration
with open("standard_unet_training_duration.txt", "w") as file:
    file.write(str(duration))

# Plot and save fitting curves
loss_values = history.history['loss']
val_loss_values = history.history["val_loss"]
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.plot(epochs, val_loss_values, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

savefig("standard_unet_fitting_curve.png")