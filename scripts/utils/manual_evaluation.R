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
val_df[, hack:="A: Results of the 5-fold Cross-Validation"]

# Plot scoring figure
figA <- ggplot(val_df, aes(class, dice, fill=class)) +
  geom_boxplot() +
  facet_wrap(hack ~ .) + 
  scale_x_discrete(labels=c("Lungs","COVID-19 Lesion")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") +
  labs(x = "", y="Dice Similarity Coefficient")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/validation_boxplot.png", width=700, height=600, res=180)
figA
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
  scale_x_discrete(labels=c("Lungs","COVID-19 Lesion")) +
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
  scale_x_discrete(labels=c("Lungs","COVID-19 Lesion")) +
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
  scale_x_discrete(labels=c("Lungs","COVID-19 Lesion")) +
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
  scale_color_discrete(name="Annotation", labels=c("Lungs", "COVID-19 Lesion")) +
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
  scale_color_discrete(name="Annotation", labels=c("Lungs", "COVID-19 Lesion")) +
  labs(x="Dice Similarity Coefficient", y="Accuracy") +
  ggtitle("5-fold CV Results - Acc vs DSC")
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/acc_vs_dsc.png", width=1000, height=800, res=200)
plot_score
dev.off()


###########################################################################################
# REWORKED: Figure 4
###########################################################################################

# Load cv5 data
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/cv_results.detailed.csv"
validation <- fread(path, sep=",", header=TRUE)
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
df_cv5 <- melt(validation,
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)
df_cv5[, "cv":="k=5"]

# Load cv4 data
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.cv4/cv_results.detailed.csv"
validation <- fread(path, sep=",", header=TRUE)
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
df_cv4 <- melt(validation,
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)
df_cv4[, "cv":="k=4"]

# Load cv3 data
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.cv3/cv_results.detailed.csv"
validation <- fread(path, sep=",", header=TRUE)
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
df_cv3 <- melt(validation,
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)
df_cv3[, "cv":="k=3"]

# Load cv2 data
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.cv2/cv_results.detailed.csv"
validation <- fread(path, sep=",", header=TRUE)
validation[, lungs:=rowMeans(validation[,c("lung_R", "lung_L")])]
validation <- validation[, c("index", "score", "lungs", "infection")]
df_cv2 <- melt(validation,
               measure.vars=c("infection", "lungs"),
               variable.name="class",
               value.name="value",
               variable.factor=TRUE)
df_cv2[, "cv":="k=2"]

# Combine all cv-N data
df <- rbindlist(list(df_cv2, df_cv3, df_cv4, df_cv5))

# Preprocess
df_filtered <- df[df$class=="infection" & df$score=="DSC"]
df_filtered$class <- "B: COVID-19 Lesion - Ma et al. Dataset"
df_filtered_mean <- df_filtered[, .(mean=mean(value)), by=cv]

# Plot Figure 4-B
figB <- ggplot(data=df_filtered, aes(cv, value)) +
  geom_bar(data=df_filtered_mean, aes(cv, mean, fill=cv), col="black",
           alpha=0.4, stat="identity", position="stack", width=0.5) +
  stat_boxplot(geom ='errorbar', width = 0.2) +
  geom_boxplot(aes(fill=cv), width=0.3) +
  facet_wrap(class ~ .) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
  coord_flip() +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  scale_color_brewer(palette="Dark2") +
  theme(legend.position = "none") +
  labs(x = "k-fold Cross-Validation", y="Dice Similarity Coefficient")

###########################################################################################

# Read testing data
path <- "/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation.testing/"
files <- list.files(path, full.names=TRUE, include.dirs=TRUE, recursive=FALSE)
# Iterate over all files
dt_list <- list()
for(file in files){
  # Skip wayne files
  if (substring(file, nchar(file)-7) == ".std.csv"){ next }
  if (substring(file, nchar(file)-8) == ".mean.csv"){ next }
  if (substring(file, nchar(file)-10) == ".median.csv"){ next }
  # identify CV and fold
  cv <- substring(file, nchar(file)-8, nchar(file)-7)
  f <- substring(file, nchar(file)-5, nchar(file)-4)
  # read dataset
  dt_tmp <- fread(file, sep=",", header=TRUE)
  dt_tmp[, "fold":=f]
  dt_tmp[, "cv":=cv]
  # append to list
  dt_list[[file]] <- dt_tmp
}

# Combine list to single datatable
dt <- rbindlist(dt_list)
# Melt
dt <- melt(dt, measure.vars=c("background", "infection"),
           variable.name="class",
           value.name="value",
           variable.factor=TRUE)

# Plot Figure 4-C
dt_filtered <- dt[dt$score=="DSC" & dt$class=="infection"]
dt_filtered$class <- "C: COVID-19 Lesion - An et al. Dataset"
dt_filtered <- dt_filtered[index %in% sort(dt_filtered$index)[1:100]]
figC <- ggplot(dt_filtered, aes(cv, value, fill=fold)) +
  geom_boxplot() +
  facet_wrap(class ~ .) +
  scale_x_discrete(labels=c("k=2", "k=3", "k=4", "k=5")) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0, 1)) +
  coord_flip() +
  theme_bw() +
  scale_fill_brewer(palette="Dark2") +
  theme(legend.position = "none") +
  labs(x = "k-fold Cross-Validation", y="Dice Similarity Coefficient")

###########################################################################################

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/figure4.2000x800.png", width=2000, height=800, res=170)
multiplot(figA, figB, figC, layout=matrix(c(1,2,2,1,3,3), nrow=2, ncol=3, byrow=TRUE))
dev.off()
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/figure4.1800x800.png", width=1800, height=800, res=170)
multiplot(figA, figB, figC, layout=matrix(c(1,2,2,1,3,3), nrow=2, ncol=3, byrow=TRUE))
dev.off()
png("/home/mudomini/projects/covid19.MIScnn.RESULTS/evaluation/figure4.1800x800.SR.png", width=1800, height=800, res=150)
multiplot(figA, figB, figC, layout=matrix(c(1,2,2,1,3,3), nrow=2, ncol=3, byrow=TRUE))
dev.off()

###########################################################################################
# Summary An et al.
dt <- dt[index %in% sort(dt$index)[1:100]]
summary <- dt[, .(median=median(value), std=sd(value)), by=c("score", "cv", "class")]
summary <- summary[summary$class=="infection"]
summary
# Summary Ma et al.
summary <- df[, .(median=median(value), std=sd(value)), by=c("score", "cv", "class")]
summary
