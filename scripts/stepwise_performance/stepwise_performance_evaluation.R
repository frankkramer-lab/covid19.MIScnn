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
library("ggplot2")
library("data.table")
library("magrittr")

# Data path
path_all <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/"
path_noDAnoPP <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noDA_noPP/"
path_noDA <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noDA/"
path_noPP <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noPP/"

###################################################################################################################
# Performance Analysis #
###################################################################################################################

# Load data
res_wDwP <- fread(file.path(path_all, "cv_results.detailed.csv"), sep=",", header=TRUE)
res_nDnP <- fread(file.path(path_noDAnoPP, "cv_results.detailed.csv"), sep=",", header=TRUE)
res_nDwP <- fread(file.path(path_noDA, "cv_results.detailed.csv"), sep=",", header=TRUE)
res_wDnP <- fread(file.path(path_noPP, "cv_results.detailed.csv"), sep=",", header=TRUE)

# Add mapping column
res_wDwP[, method := "wDwP"]
res_nDnP[, method := "nDnP"]
res_nDwP[, method := "nDwP"]
res_wDnP[, method := "wDnP"]

# Combine tables
res <- rbind(res_wDwP, res_nDnP, res_nDwP, res_wDnP)

# Preprocessing
res[, lungs:=rowMeans(res[,c("lung_R", "lung_L")])]
res <- res[, c("index", "method", "score", "lungs", "infection")]
res_df <- melt(res,
               measure.vars=c("lungs", "infection"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)

# Reorder classes
res_df$method <- factor(res_df$method , levels=c("nDnP", "wDnP", "nDwP", "wDwP"))
res_df[method=="nDnP"]$method <- "Data Aug: Excluded & PreProc: Excluded"
res_df[method=="wDnP"]$method <- "Data Aug: Included & PreProc: Excluded"
res_df[method=="nDwP"]$method <- "Data Aug: Excluded & PreProc: Included"
res_df[method=="wDwP"]$method <- "Data Aug: Included & PreProc: Included"
res_df[score=="Acc"]$score <- "Accuracy"
res_df[score=="DSC"]$score <- "Dice Similarity Coef."
res_df[score=="Sens"]$score <- "Sensitivity"
res_df[score=="Spec"]$score <- "Specificity"
res_df[class=="lungs"]$class <- "Lungs"
res_df[class=="infection"]$class <- "COVID-19"
res_df$class <- factor(res_df$class , levels=c("Lungs","COVID-19"))

# Plot scoring figure for histograms
plot_score <- ggplot(res_df[score=="Dice Similarity Coef."], aes(value)) +
  geom_histogram(binwidth = 0.05, color="darkblue", fill="lightblue") +
  facet_grid(class ~ method) +
  scale_y_continuous(breaks=seq(0, 20, 4), limits=c(0, 20)) +
  #scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") +
  labs(x = "Dice Similarity Coefficient", y="Sample Frequency") +
  ggtitle("Stepwise Pipeline Performance Evaluation: DSC Comparision ")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/spe.histogram.png", width=2000, height=800, res=180)
plot_score
dev.off()


# Plot scoring figure for boxplots
res_df$class <- factor(res_df$class , levels=c("COVID-19","Lungs"))
plot_score <- ggplot(res_df, aes(class, value, fill=class)) +
  geom_boxplot() +
  coord_flip() +
  facet_grid(score ~ method) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") +
  labs(x="", y="Metric Score") +
  ggtitle("Stepwise Pipeline Performance Evaluation: All Metrics")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/spe.boxplot.png", width=2000, height=1200, res=180)
plot_score
dev.off()
