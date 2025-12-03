Overview:

This project implements a Temporal Convolutional Network (TCN) combined with a Self-Attention mechanism for time series forecasting. It also integrates Gaussian Process Regression (GPR) to analyze prediction uncertainty and local variability. The system supports automatic hyperparameter optimization (Optuna), model training, validation, testing, and visualization.

Requirements:

pip install torch numpy pandas matplotlib optuna scikit-learn scipy tqdm openpyxl


Usage:

1、Train Model & Optimize Hyperparameters
python TCN_SA_Guss_opti.py
- Automatically splits data and trains the model
- Runs Optuna optimization (default: 70 trials)
- Saves results to trialresult/ and Best_trial.csv

2、Gaussian Process Analysis
python GP.py
- Generates prediction confidence intervals and anomaly detection


Run:

python TCN_SA_Guss_opti.py 

data_path ./saltMA.csv \

target OT \


Author:

ylhuang1995@whu.edu.cn
