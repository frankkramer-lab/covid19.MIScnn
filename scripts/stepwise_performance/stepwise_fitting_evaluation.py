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
# Import libraries
import pandas as pd
import os
from plotnine import *

# Hardcoded data pathes
path_all = "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/"
path_noDAnoPP = "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noDA_noPP/"
path_noDA = "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noDA/"

# Build dataset from logging data
method = ["wDwP", "nDwP", "nDnP"]
pathes = [path_all, path_noDA, path_noDAnoPP]
dt_raw = pd.DataFrame()
for i in range(0, 3):
    dt_tmp = pd.DataFrame()
    for fold in ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]:
        path_log = os.path.join(pathes[i], fold, "history.tsv")
        fitting_log = pd.read_csv(path_log, sep="\t")
        fitting_log["fold"] = fold
        fitting_log["method"] = method[i]
        dt_tmp = dt_tmp.append(fitting_log, ignore_index=True)
    dt_raw = dt_raw.append(dt_tmp, ignore_index=True)

# Preprocessing
res_dt = pd.melt(dt_raw,
                 id_vars=["method", "fold", "epoch"],
                 value_vars=["loss", "tversky_loss", "dice_soft", "dice_crossentropy",
                             "val_loss", "val_tversky_loss", "val_dice_soft", "val_dice_crossentropy"],
                 var_name="metric", value_name="value")

# Renaming stuff
res_dt.replace({"nDnP": "Data Augmentation: Excluded & Preprocessing: Excluded",
                "nDwP": "Data Augmentation: Excluded & Preprocessing: Included",
                "wDwP": "Data Augmentation: Included & Preprocessing: Included"},
                inplace=True)

# Metric selection - loss
loss_dt = res_dt[(res_dt["metric"]=="loss") | (res_dt["metric"]=="val_loss")]

# Plot fitting curve
fig = (ggplot(loss_dt, aes("epoch", "value", color="metric", group="metric"))
              + geom_smooth(method="gpr", size=1)
              + facet_wrap("method", ncol=1)
              + scale_y_continuous(limits=[0, 5])
              + ggtitle("Stepwise Fitting Curve Evaluation")
              + xlab("Number of Epochs")
              + ylab("Loss Function")
              + scale_colour_discrete(name="Dataset",
                                      labels=["Training", "Validation"])
              + theme_bw(base_size=28))
fig.save(filename="spe.fitting_curve.png", path="evaluation",
         width=20, height=16, dpi=200)
