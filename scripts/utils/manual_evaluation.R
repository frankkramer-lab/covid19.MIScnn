# Import libraries
library("ggplot2")
library("data.table")
library("magrittr")

# Data path
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/cv_results.detailed.csv"

###########################################################################################
# Dice
###########################################################################################

# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[score=="DSC", c("index", "lungs", "infection")]
val_df <- melt(validation, 
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="dice",
               variable.factor=TRUE)

# Reorder classes
val_df$class <- factor(val_df$class , levels=c("lungs","infection"))

# Plot scoring figure
plot_score <- ggplot(val_df, aes(class, dice, fill=class)) + 
  geom_boxplot() +
  scale_x_discrete(labels=c("Lungs","COVID-19")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "", y="Dice Similarity Coefficient") + 
  ggtitle("Results of the 5-fold Cross-Validation")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/validation_boxplot.png", width=700, height=600, res=180)
plot_score
dev.off()

###########################################################################################
# Accuracy
###########################################################################################

# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[score=="Acc", c("index", "lungs", "infection")]
val_df <- melt(validation, 
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="dice",
               variable.factor=TRUE)

# Plot scoring figure for accuracy
plot_score <- ggplot(val_df, aes(class, dice, fill=class)) + 
  geom_boxplot() +
  scale_x_discrete(labels=c("Lungs","COVID-19")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "", y="Pixelwise Accuracy") + 
  ggtitle("Results of the 5-fold Cross-Validation")
png("score.png", width=700, height=600, res=180)
plot_score
dev.off()


###########################################################################################
# Boxplot - Multiplot
###########################################################################################
# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
val_df <- melt(validation, 
               measure.vars=c("lungs", "infection"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)

# Reorder classes
val_df$class <- factor(val_df$class , levels=c("lungs","infection"))
val_df[score=="Acc"]$score <- "Accuracy"
val_df[score=="DSC"]$score <- "Dice Similarity Coef."
val_df[score=="Sens"]$score <- "Sensitivity"
val_df[score=="Spec"]$score <- "Specificity"
val_df[score=="Prec"]$score <- "Precision"
val_df[score=="IoU"]$score <- "Intersection over Union"

# Plot scoring figure for accuracy
plot_score <- ggplot(val_df, aes(class, value, fill=class)) + 
  geom_boxplot() +
  facet_grid(score ~ .) +
  scale_x_discrete(labels=c("Lungs","COVID-19")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "", y="") + 
  ggtitle("Results of the 5-fold Cross-Validation")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.boxplot.png", width=800, height=1200, res=180)
plot_score
dev.off()


###########################################################################################
# Histogram - Multiplot
###########################################################################################
# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
val_df <- melt(validation, 
               measure.vars=c("lungs", "infection"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)

# Reorder classes
val_df[score=="Acc"]$score <- "Accuracy"
val_df[score=="DSC"]$score <- "Dice Similarity Coef."
val_df[score=="Sens"]$score <- "Sensitivity"
val_df[score=="Spec"]$score <- "Specificity"
val_df[score=="Prec"]$score <- "Precision"
val_df[score=="IoU"]$score <- "Intersection over Union"
val_df[class=="lungs"]$class <- "Lungs"
val_df[class=="infection"]$class <- "COVID-19"
val_df$class <- factor(val_df$class , levels=c("Lungs","COVID-19"))

# Plot scoring figure for boxplots
plot_score <- ggplot(val_df, aes(value)) + 
  geom_histogram(binwidth = 0.05, color="darkblue", fill="lightblue") +
  facet_grid(score ~ class) +
  scale_y_continuous(breaks=seq(0, 20, 4), limits=c(0, 20)) + 
  #scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "Metric Score", y="Sample Frequency") + 
  ggtitle("Results of the 5-fold Cross-Validation")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.histogram.png", width=1000, height=1600, res=160)
plot_score
dev.off()

png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.histogram.large.png", width=1400, height=1800, res=180)
plot_score
dev.off()


###########################################################################################
# Boxplot - Multiplot
###########################################################################################
# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
val_df <- melt(validation, 
               measure.vars=c("lungs", "infection"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)

# Reorder classes
val_df$class <- factor(val_df$class , levels=c("lungs","infection"))
val_df[score=="Acc"]$score <- "Accuracy"
val_df[score=="DSC"]$score <- "Dice Similarity Coef."
val_df[score=="Sens"]$score <- "Sensitivity"
val_df[score=="Spec"]$score <- "Specificity"
val_df[score=="Prec"]$score <- "Precision"
val_df[score=="IoU"]$score <- "Intersection over Union"

# Plot scoring figure for histograms
plot_score <- ggplot(val_df, aes(class, value, fill=class)) + 
  geom_boxplot() +
  facet_grid(score ~ .) +
  scale_x_discrete(labels=c("Lungs","COVID-19")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") + 
  labs(x = "", y="") + 
  ggtitle("Results of the 5-fold Cross-Validation")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/multiplot.boxplot.png", width=800, height=1400, res=140)
plot_score
dev.off()


###########################################################################################
# Sensitivity vs DSC vs Accuracy
###########################################################################################
# Load data
validation <- fread(path, sep=",", header=TRUE)

# Preprocessing
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
validation <- melt(validation, 
                   measure.vars=c("lungs", "infection"),
                   variable.name="class",
                   value.name="value",
                   variable.factor=TRUE)
val_df <- dcast(validation, index + class ~ score, value.var=c("value"))

# Plot scoring figure for SENS vs DSC
plot_score <- ggplot(val_df, aes(DSC, Sens, col=class)) + 
  geom_abline(intercept=0, slope=1, size=0.1) + 
  geom_point() +
  scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_color_discrete(name="Annotation", labels=c("Lungs", "COVID-19")) + 
  labs(x="Dice Similarity Coefficient", y="Sensitivity") + 
  ggtitle("5-fold CV Results - Sens vs DSC")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/sens_vs_dsc.png", width=1000, height=800, res=200)
plot_score
dev.off()

# Plot scoring figure for DSC vs ACC
plot_score <- ggplot(val_df, aes(DSC, Acc, col=class)) + 
  geom_point() +
  scale_x_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) + 
  theme_bw() +
  scale_color_discrete(name="Annotation", labels=c("Lungs", "COVID-19")) + 
  labs(x="Dice Similarity Coefficient", y="Accuracy") + 
  ggtitle("5-fold CV Results - Acc vs DSC")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/acc_vs_dsc.png", width=1000, height=800, res=200)
plot_score
dev.off()


###########################################################################################
# SD Values
###########################################################################################