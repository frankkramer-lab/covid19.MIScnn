# Stepwise Performance Evaluation

In this additional evaluation, we try to analyze the stepwise model performance influence of the Preprocessing and Data Augmentation step in our pipeline.

We utilize the same cross-validation sampling of the run_preprocessing.py script in order to ensure comparability.

The only differences are in the run_miscnn.py scripts which, now, do NOT feature the Preprocessing & Data Aug. steps and, therefore, their code lines are excluded.

### Pipeline Run without Data Augmentation and without Preprocessing

Run the following commands:

```sh
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 0
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 1
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 2
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 3
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 4

python3 scripts/run_evaluation.py
```

Rename the resulting evaluation directory.

```sh
mv evaluation/ evaluation.stepwise.noDA_noPP/
```

### Pipeline Run without Data Augmentation and with Preprocessing

Run the following commands:

```sh
python3 scripts/stepwise_performance/run_miscnn.noDA.py --fold 0
python3 scripts/stepwise_performance/run_miscnn.noDA.py --fold 1
python3 scripts/stepwise_performance/run_miscnn.noDA.py --fold 2
python3 scripts/stepwise_performance/run_miscnn.noDA.py --fold 3
python3 scripts/stepwise_performance/run_miscnn.noDA.py --fold 4

python3 scripts/run_evaluation.py
```

Rename the resulting evaluation directory.

```sh
mv evaluation/ evaluation.stepwise.noDA/
```

### Pipeline Run with Data Augmentation and without Preprocessing

Run the following commands:

```sh
python3 scripts/stepwise_performance/run_miscnn.noPreProc.py --fold 0
python3 scripts/stepwise_performance/run_miscnn.noPreProc.py --fold 1
python3 scripts/stepwise_performance/run_miscnn.noPreProc.py --fold 2
python3 scripts/stepwise_performance/run_miscnn.noPreProc.py --fold 3
python3 scripts/stepwise_performance/run_miscnn.noPreProc.py --fold 4

python3 scripts/run_evaluation.py
```

Rename the resulting evaluation directory.

```sh
mv evaluation/ evaluation.stepwise.noPP/
```

### Evaluation

For running the stepwise performance evaluation, you have to modify the hardcoded pathes inside the following files:
- 'stepwise_fitting_evaluation.py'
- 'stepwise_performance_evaluation.py'

Afterwards, run the following commands:

```sh
Rscript scripts/stepwise_performance/stepwise_performance_evaluation.R
python3 scripts/stepwise_performance/stepwise_fitting_evaluation.py
```

### Results
