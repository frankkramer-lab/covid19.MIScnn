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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO
from miscnn.evaluation.cross_validation import load_disk2fold
from plotnine import *
import argparse

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="Automated COVID-19 Segmentation")
parser.add_argument("-p", "--predictions", help="Path to predictions directory",
                    required=True, type=str, dest="pred")
args = parser.parse_args()
pred_path = args.pred
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
eval_path = "evaluation.testing"

#-----------------------------------------------------#
#                  Score Calculations                 #
#-----------------------------------------------------#
def calc_DSC(truth, pred, classes):
    dice_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate Dice
            dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice Similarity Coefficients
    return dice_scores

def calc_IoU(truth, pred, classes):
    iou_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate iou
            iou = np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum() - np.logical_and(pd, gt).sum())
            iou_scores.append(iou)
        except ZeroDivisionError:
            iou_scores.append(0.0)
    # Return computed IoU
    return iou_scores

def calc_Sensitivity(truth, pred, classes):
    sens_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate sensitivity
            sens = np.logical_and(pd, gt).sum() / gt.sum()
            sens_scores.append(sens)
        except ZeroDivisionError:
            sens_scores.append(0.0)
    # Return computed sensitivity scores
    return sens_scores

def calc_Specificity(truth, pred, classes):
    spec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate specificity
            spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
            spec_scores.append(spec)
        except ZeroDivisionError:
            spec_scores.append(0.0)
    # Return computed specificity scores
    return spec_scores

def calc_Accuracy(truth, pred, classes):
    acc_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate accuracy
            acc = (np.logical_and(pd, gt).sum() + \
                   np.logical_and(not_pd, not_gt).sum()) /  gt.size
            acc_scores.append(acc)
        except ZeroDivisionError:
            acc_scores.append(0.0)
    # Return computed accuracy scores
    return acc_scores

def calc_Precision(truth, pred, classes):
    prec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate precision
            prec = np.logical_and(pd, gt).sum() / pd.sum()
            prec_scores.append(prec)
        except ZeroDivisionError:
            prec_scores.append(0.0)
    # Return computed precision scores
    return prec_scores

#-----------------------------------------------------#
#                    Run Evaluation                   #
#-----------------------------------------------------#
# Initialize Data IO Interface for NIfTI data
## We are using 4 classes due to [background, lung_left, lung_right, covid-19]
interface = NIFTI_interface(channels=1, classes=4)

# Create Data IO object to load and write samples in the file structure
data_io = Data_IO(interface, input_path="data.testing", output_path=pred_path)

# Access all available samples in our file structure
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Initialize dataframe
cols = ["index", "score", "background", "infection"]
df = pd.DataFrame(data=[], dtype=np.float64, columns=cols)

# Iterate over each sample
for index in tqdm(sample_list):
    # Load a sample including its image, ground truth and prediction
    sample = data_io.sample_loader(index, load_seg=True, load_pred=True)
    # Access image, ground truth and prediction data
    image = sample.img_data
    truth = sample.seg_data
    pred = sample.pred_data

    pred = np.where(pred==1, 0, pred)
    pred = np.where(pred==2, 0, pred)
    pred = np.where(pred==3, 1, pred)

    # Compute diverse Scores
    dsc = calc_DSC(truth, pred, classes=2)
    df = df.append(pd.Series([index, "DSC"] + dsc, index=cols),
                   ignore_index=True)
    iou = calc_IoU(truth, pred, classes=2)
    df = df.append(pd.Series([index, "IoU"] + iou, index=cols),
                   ignore_index=True)
    sens = calc_Sensitivity(truth, pred, classes=2)
    df = df.append(pd.Series([index, "Sens"] + sens, index=cols),
                   ignore_index=True)
    spec = calc_Specificity(truth, pred, classes=2)
    df = df.append(pd.Series([index, "Spec"] + spec, index=cols),
                   ignore_index=True)
    prec = calc_Precision(truth, pred, classes=2)
    df = df.append(pd.Series([index, "Prec"] + prec, index=cols),
                   ignore_index=True)
    acc = calc_Accuracy(truth, pred, classes=2)
    df = df.append(pd.Series([index, "Acc"] + acc, index=cols),
                   ignore_index=True)

# Output complete dataframe
print(df)
# Create evaluation directory
if not os.path.exists(eval_path) : os.mkdir(eval_path)
# Identify cv & fold
id = pred_path.split(".")[-1]
# Store complete dataframe to disk
path_res_detailed = os.path.join(eval_path, "results." + id + ".csv")
df.to_csv(path_res_detailed, index=False)

# Print out average, median std evaluation metrics for the current fold
df_avg = df.groupby(by="score").mean()
path_out = os.path.join(eval_path, "results." + id + ".mean.csv")
df_avg.to_csv(path_out, index=True)
df_med = df.groupby(by="score").median()
path_out = os.path.join(eval_path, "results." + id + ".median.csv")
df_med.to_csv(path_out, index=True)
df_std = df.groupby(by="score").std()
path_out = os.path.join(eval_path, "results." + id + ".std.csv")
df_std.to_csv(path_out, index=True)
