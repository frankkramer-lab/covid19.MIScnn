#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import tensorflow as tf
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, \
                                          dice_crossentropy, tversky_loss
from miscnn.evaluation.cross_validation import cross_validation
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                       EarlyStopping, CSVLogger
from miscnn.evaluation.cross_validation import run_fold, load_csv2fold
import argparse
import os

#-----------------------------------------------------#
#      Tensorflow Configuration for GPU Cluster       #
#-----------------------------------------------------#
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Automated COVID-19 Segmentation")
parser.add_argument("-f", "--fold", help="Cross-validation fold. Range: [0:5]",
                    required=True, type=int, dest="fold")
args = parser.parse_args()
fold = args.fold
fold_subdir = os.path.join("evaluation", "fold_" + str(fold))

#-----------------------------------------------------#
#               Setup of MIScnn Pipeline              #
#-----------------------------------------------------#
# Initialize Data IO Interface for NIfTI data
## We are using 4 classes due to [background, lung_left, lung_right, covid-19]
interface = NIFTI_interface(channels=1, classes=4)

# Create Data IO object to load and write samples in the file structure
data_io = Data_IO(interface, input_path="data", delete_batchDir=False)

# Access all available samples in our file structure
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True,
                             gaussian_noise=True)

# Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
sf_clipping = Clipping(min=-1250, max=250)
# Create a pixel value normalization Subfunction to scale between 0-255
sf_normalize = Normalization(mode="grayscale")
# Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
sf_resample = Resampling((1.58, 1.58, 2.70))
# Create a pixel value normalization Subfunction for z-score scaling
sf_zscore = Normalization(mode="z-score")

# Assemble Subfunction classes into a list
sf = [sf_clipping, sf_normalize, sf_resample, sf_zscore]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="patchwise-crop", patch_shape=(160, 160, 80))
# Adjust the patch overlap for predictions
pp.patchwise_overlap = (80, 80, 40)

# Initialize the Architecture
unet_standard = Architecture(depth=4, activation="softmax",
                             batch_normalization=True)

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, architecture=unet_standard,
                       loss=tversky_crossentropy,
                       metrics=[tversky_loss, dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=3, learninig_rate=0.001)

# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=15,
                          verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
cb_es = EarlyStopping(monitor="loss", patience=100)
cb_tb = TensorBoard(log_dir=os.path.join(fold_subdir, "tensorboard"),
                    histogram_freq=0, write_graph=True, write_images=True)
cb_cl = CSVLogger(os.path.join(fold_subdir, "logs.csv"), separator=',',
                  append=True)

#-----------------------------------------------------#
#          Run Pipeline for provided CV Fold          #
#-----------------------------------------------------#
# Run pipeline for cross-validation fold
run_fold(fold, model, epochs=1000, iterations=150, evaluation_path="evaluation",
         draw_figures=True, callbacks=[cb_lr, cb_es, cb_tb, cb_cl],
         save_models=False)

# Dump model to disk for reproducibility
model.dump(os.path.join(fold_subdir, "model.hdf5"))

#-----------------------------------------------------#
#           Inference for provided CV Fold            #
#-----------------------------------------------------#
# Obtain training and validation data set
training, validation = load_csv2fold(os.path.join(fold_subdir,
                                                  "sample_list.csv"))
# Compute predictions
model.predict(validation, direct_output=False)
