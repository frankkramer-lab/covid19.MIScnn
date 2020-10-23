# Stepwise Performance Evaluation

In this additional evaluation, we try to analyze the stepwise model performance influence of the Preprocessing and Data Augmentation step in our pipeline.

We utilize the same cross-validation sampling of the run_preprocessing.py script in order to ensure comparability.

The only differences are in the run_miscnn.py scripts which, now, do NOT feature the Preprocessing & Data Aug. steps and, therefore, their codelines are excluded.

### Pipeline Run without Data Augmentation and without Preprocessing

Run the following commands:

```sh
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 0
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 1
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 2
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 3
python3 scripts/stepwise_performance/run_miscnn.noDA_noPreProc.py --fold 4
```
