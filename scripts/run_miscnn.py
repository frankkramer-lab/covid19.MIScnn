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
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

#-----------------------------------------------------#
#           Custom Normalization Subfunction          #
#-----------------------------------------------------#
class Normalization_Grayscale(Abstract_Subfunction):
    def __init__(self):
        pass

    def preprocessing(self, sample, training=True):
        # Access image
        image = sample.img_data
        # Perform normalization
        image_normalized = (image - 101) / 76.9
        # Update the sample with the normalized image
        sample.img_data = image_normalized

    def postprocessing(self, prediction):
        return prediction

#-----------------------------------------------------#
#             Running the MIScnn Pipeline             #
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
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True,
                             gaussian_noise=True)

# Create a clipping Subfunction to the lung window of CTs (-1250 and 250)
sf_clipping = Clipping(min=-1250, max=250)
# Create a pixel value normalization Subfunction to scale 0-1
sf_normalize = Normalization(z_score=False)
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((1.62, 1.62, 3.22))

# Assemble Subfunction classes into a list
sf = [sf_clipping, sf_normalize]

#sf = [Resampling((9,9,9)), Padding((32,32,32), shape_must_be_divisible_by=16)]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage", patch_shape=(80, 160, 160))
# Adjust the patch overlap for predictions
pp.patchwise_overlap = (40, 80, 80)



# Initialize Keras Data Generator for generating batches
from miscnn.neural_network.data_generator import DataGenerator
dataGen = DataGenerator(sample_list, pp, training=True, validation=False,
                        shuffle=False)

for img, seg in dataGen:
    print(img.shape)
    print(img.min(), img.max())
    #print(batch.shape)

#
#
# # Library import
# from miscnn.neural_network.model import Neural_Network
# from miscnn.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss
#
# # Create the Neural Network model
# model = Neural_Network(preprocessor=pp, loss=tversky_loss, metrics=[dice_soft, dice_crossentropy],
#                        batch_queue_size=3, workers=3, learninig_rate=0.0001)


## In COVID-19-CT-Seg dataset, the last 10 cases from Radiopaedia have been
## adjusted to lung window [-1250,250], and then normalized to [0,255],
## we recommend to adust the first 10 cases from Coronacases with the same method.
