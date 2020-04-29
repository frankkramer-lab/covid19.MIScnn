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
from miscnn import Data_IO
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"

#-----------------------------------------------------#
#                   Data Exploration                  #
#-----------------------------------------------------#
# Initialize Data IO Interface for NIfTI data
interface = NIFTI_interface(channels=1, classes=3)

# Create Data IO object to load and write samples in the file structure
data_io = Data_IO(interface, path_data, delete_batchDir=True)

# Access all available samples in our file structure
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Print out the sample list
print("Sample list:", sample_list)

# Now let's load each sample and obtain collect diverse information from them
sample_data = {}
for index in tqdm(sample_list):
    # Sample loading
    sample = data_io.sample_loader(index, load_seg=True)
    # Create an empty list for the current asmple in our data dictionary
    sample_data[index] = []
    # Store the volume shape
    sample_data[index].append(sample.img_data.shape)
    # Identify minimum and maximum volume intensity
    sample_data[index].append(sample.img_data.min())
    sample_data[index].append(sample.img_data.max())
    # Store voxel spacing
    sample_data[index].append(sample.details["spacing"])
    # Identify and store class distribution
    unique_data, unique_counts = np.unique(sample.seg_data, return_counts=True)
    class_freq = unique_counts / np.sum(unique_counts)
    class_freq = np.around(class_freq, decimals=6)
    sample_data[index].append(tuple(class_freq))

# Transform collected data into a pandas dataframe
df = pd.DataFrame.from_dict(sample_data, orient="index",
                            columns=["vol_shape", "vol_minimum",
                                     "vol_maximum", "voxel_spacing",
                                     "class_frequency"])

# Print out the dataframe to console
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

# Calculate mean and median shape sizes
shape_list = np.array(df["vol_shape"].tolist())
for i, a in enumerate(["X", "Y", "Z"]):
    print(a + "-Axes Mean:", np.mean(shape_list[:,i]))
    print(a + "-Axes Median:", np.median(shape_list[:,i]))

## The resolution of the volumes are fixed for the x,y axes to 512x512 for the
## "coronacases" data and 630x630 for the "radiopaedia" volumes.
## Strangely, a single sample "radiopaedia_14_85914_0" has only 630x401 for x,y.
## Furthermore, the sample "radiopaedia_10_85902_3" has a ~ 8x times higher
## z axis than the other radiopaedia samples.
## Overall, the aim to identify suited resampling coordinates will be tricky.
##
## We see that the radiopaedia ct volumes have already preprocessed
## In the description of the data set, it is documentated that they
## used a clipping to the lung window [-1250, 250] and normalized these values
## to [0,255]. Therefore we have to perform the identical preprocessing to the
## other ct volumes as well
##
## Also, the class frequency reveal a heavy bias towards the background class
## as expected in medical image segmentation
##
## In COVID-19-CT-Seg dataset, the last 10 cases from Radiopaedia have been
## adjusted to lung window [-1250,250], and then normalized to [0,255],
## we recommend to adust the first 10 cases from Coronacases with the same method.
