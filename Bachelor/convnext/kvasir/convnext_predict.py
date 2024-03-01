# import libraries
import numpy as np
from miscnn.data_loading.interfaces import Image_interface
from miscnn.data_loading.data_io import Data_IO
from miscnn.processing.preprocessor import Preprocessor
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction
from miscnn.processing.subfunctions import Resize, Normalization
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss

from convnext_unet import Architecture as Convnext

# ----------------------------   Paths   ------------------------------------ # 
# Change these paths to where the corresponding data is stored on your system #
# --------------------------------------------------------------------------- #
# Path to test subset of Kvasir-SEG dataset
data_path = "jonas/Bachelor/kvasir_seg/test/data/"
# Path to the saved model
model_path = "jonas/Bachelor/convnext/convnext_unet_kvasir.hdf5"


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
        sample.seg_data = np.where(seg_temp != 0, 1, seg_temp)
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
sf = [sf_resize, sf_change, sf_zscore]

# Initialize Data_IO
interface = Image_interface(classes=2, img_type="rgb", img_format="png")
data_io = Data_IO(interface, data_path, delete_batchDir=True)

# Create Preprocessor (takes Data_IO as argument)
pp = Preprocessor(data_io, batch_size=5, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage")

# Initialize architecture and neural network
convnext = Convnext()
convnext_unet = Neural_Network(preprocessor=pp, loss=dice_soft, architecture=convnext,
                       metrics=[tversky_loss, dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=5, learning_rate=0.001)

# Load the saved weights
convnext_unet.load(model_path)
test = data_io.get_indiceslist()

# Generate predictions
convnext_unet.predict(test)