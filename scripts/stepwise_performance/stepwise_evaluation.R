# Import libraries
library("ggplot2")
library("data.table")
library("magrittr")

# Data path
path_all <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/cv_results.detailed.csv"
path_noDAnoPP <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.stepwise.noDA_noPP/cv_results.detailed.csv"

# Load data
res_wDwP <- fread(path_all, sep=",", header=TRUE)
res_nDnP <- fread(path_noDAnoPP, sep=",", header=TRUE)

# Add mapping column
res_wDwP[, method := "wDwP"]
res_nDnP[, method := "nDnP"]
#res_wDwP[, method := "wDwP"]
#res_wDwP[, method := "wDwP"]

# Combine tables
res <- rbind(res_wDwP, res_nDnP)

# Preprocessing
res[, lungs:=rowMeans(res[,c("lung_R", "lung_L")])]
res <- res[, c("index", "method", "score", "lungs", "infection")]
res_df <- melt(res, 
               measure.vars=c("lungs", "infection"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)

# Reorder classes
res_df[method=="nDnP"]$method <- "Data Augmentation: Excluded & Preprocessing: Excluded"
res_df[method=="wDwP"]$method <- "Data Augmentation: Included & Preprocessing: Included"
res_df[score=="Acc"]$score <- "Accuracy"
res_df[score=="DSC"]$score <- "Dice Similarity Coef."
res_df[score=="Sens"]$score <- "Sensitivity"
res_df[score=="Spec"]$score <- "Specificity"
res_df[class=="lungs"]$class <- "Lungs"
res_df[class=="infection"]$class <- "COVID-19"
res_df$class <- factor(res_df$class , levels=c("Lungs","COVID-19"))

# Plot scoring figure for histograms
ggplot(res_df[score=="Dice Similarity Coef."], aes(value)) + 
  geom_histogram(binwidth = 0.05, color="darkblue", fill="lightblue") +
  facet_grid(class ~ method) +
  scale_y_continuous(breaks=seq(0, 20, 4), limits=c(0, 20)) + 
  #scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "Dice Similarity Coefficient", y="Sample Frequency") + 
  ggtitle("Stepwise Pipeline Performance Evaluation: DSC Comparision ")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.histogram.png", width=1000, height=1200, res=180)
plot_score
dev.off()


# Plot scoring figure for boxplots
res_df$class <- factor(res_df$class , levels=c("COVID-19","Lungs"))
ggplot(res_df, aes(class, value, fill=class)) + 
  geom_boxplot() +
  coord_flip() + 
  facet_grid(score ~ method) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x="", y="Metric Score") + 
  ggtitle("Stepwise Pipeline Performance Evaluation: All Metrics")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.histogram.png", width=1000, height=1200, res=180)
plot_score
dev.off()