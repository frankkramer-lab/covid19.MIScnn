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
import requests
from tqdm import tqdm
import os
import zipfile

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"
# Links to the data set
url_vol = "https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1"
url_seg = "https://zenodo.org/record/3757476/files/Lung_and_Infection_Mask.zip?download=1"

#-----------------------------------------------------#
#                  Download Function                  #
#-----------------------------------------------------#
# Author: Shenghan Gao (wy193777)
# Modifications: MCrazy
# Source: https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2
def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        print("WARNING: Skipping download due to files are already there.")
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

#-----------------------------------------------------#
#                     Runner Code                     #
#-----------------------------------------------------#
# Create data structure
if not os.path.exists(path_data) : os.makedirs(path_data)

# Download CT volumes and save them into the data directory
path_vol_zip = os.path.join(path_data, "volumes.zip")
print("INFO:", "Downloading Volumes")
download_from_url(url_vol, path_vol_zip)
# Download segmentations and save them into the data directory
path_seg_zip = os.path.join(path_data, "segmentations.zip")
print("INFO:", "Downloading Segmentations")
download_from_url(url_seg, path_seg_zip)

# Extract sample list from the ZIP file
print("INFO:", "Obtain sample list from the volumes ZIP file")
with zipfile.ZipFile(path_vol_zip, "r") as zip_vol:
    sample_list = zip_vol.namelist()

# Iterate over the sample list and extract each sample from the ZIP files
print("INFO:", "Extracting data from ZIP files")
for sample in tqdm(sample_list):
    # Skip if file does not end with nii.gz
    if not sample.endswith(".nii.gz") : continue
    # Create sample directory
    path_sample = os.path.join(path_data, sample[:-len(".nii.gz")])
    if not os.path.exists(path_sample) : os.makedirs(path_sample)
    # Extract volume and store file into the sample directory
    with zipfile.ZipFile(path_vol_zip, "r") as zip_vol:
        zip_vol.extract(sample, path_sample)
    os.rename(os.path.join(path_sample, sample),
              os.path.join(path_sample, "imaging.nii.gz"))
    # Extract segmentation and store file into the sample directory
    with zipfile.ZipFile(path_seg_zip, "r") as zip_seg:
        zip_seg.extract(sample, path_sample)
    os.rename(os.path.join(path_sample, sample),
              os.path.join(path_sample, "segmentation.nii.gz"))

# Remove ZIP files due to disk space reduction
os.remove(path_vol_zip)
os.remove(path_seg_zip)

# Final info to console
print("INFO:", "Finished file structure creation")
