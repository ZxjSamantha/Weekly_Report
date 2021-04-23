# Pipeline of train_decoders.py

## Purpose: train and fine-tune decoders

1. Data load 

2. Tailored decoders parameters (within participant) (_tail)

n_folds: number of folds for each participant

spec_meas: specific measurements? power, power_log, relative_power, phase, freqslide

hyps: hyperparameters of tailored model
F1, dropoutRate, kernLength, kernLength_sep, dropoutType, D, n_estimators, max_depth

epochs: training epochs

patience: ?

3. Same modality but across participants decoder parameters

n_folds: number of folds for total folds 

spec_meas: specific measurements? power, power_log, relative_power, phase, freqslide

hyps: hyperparameters of cross-subjects-same-modality decoder

epochs: training epochs

4. Unseen modality testing parameters (across participants) 

eeg_roi_proj_lp: projection matrix, **not found**

5. Fine-tune parameters for same modality

model_type: select which model? 

layers: which layers to fine-tune or entire model to be retrained 

sp: output path of retrained layers 

train/validation split: use_per_vals, if True use percentage values (otherwise, use number of trials)

percentage of train/val: [0.17， 0.33， 0.5， 0.67], [0.08, 0.17, 0.25, 0.33]

number of trials train/val: [16, 34, 66, 100], [8, 16, 34, 50]

6. Train some modality decoders with different numbers of training participants 

**User-defined parameters**: max_train participants, number of validation participants

7. Tailored decoder training: 

parameters: spec_meas:[power, power_log, relative_power, phase, freqslide]

parameters of run_nn_models: 

  single_sp: rootpath + dataset+ '/single_objs'
  
  n_folds: number of folds
  
  **combined_sbjs: bool**
  
  ecog_lp: rootpath + 'ecog_dataset/'
  
  **ecog_roi_proj_lp: projection matrix file**
  
  **test_day:? the date of experiment?**
  
  **do_log: bool, only True when val == 'power_log'**
  
  epochs: training epochs
  
  **patience: ?**
  
  models: which model to select 
  
  **compute_val: ='power' if val == 'power_log'**
  
  F1:
  
  dropoutrate:
  
  kernLength:
  
  kernLength: 
  
  dropoutType:
  
  D:
  
  F2: 
  
  n_estimators






