# Small-sample learning? Zero-shot learning? 

# Source code reading of EEGNet

2 key questions:

What is the impact of ConvNet design choices on the decoding accuracies? 

The overall network architecture, the type of non-linearity 

What is the impact of ConvNet training strategies on decoding accuracies? 

training on entire trials or crops within trials. 

Filter Bank Common Spatial Patterns (FBCSP) is a method that is widely used in EEG decoding and has won several EEG decoding competitions such as BCI Competition IV 2a and 2b. 

2 sampling strategies for the deep and shallow ConvNets:

training on whole trials or on multiple crops of the trial 

using multiple crops holds promise as it increases the amount of training examples, which has been crucial to the success of deep ConvNets. 

2 methods for feature visualization that we used to gain insights into our ConvNet learned from the neuronal data. 

**EEG band power features** has been concentrated as a target of visualizations. 

Band power features should be discriminative for the different classes. FBCSP uses these features too. 

1. How much information about a specific feature is retained in the ConvNet in different layers. 

2. Investigate causal effects of the feature values on the ConvNet outputs. 

Contributions:

1. **batch normalization** and **exponential linear units** are cruicial for reaching high decoding accuracies. 

2. cropped training can increase the decoding accuracy

one EEG data set per subject $i$ 

benchmark of this work: FBCSP

workflow:

1. Bandpass filtering: different bandpass filters are applied to separate the raw EEG signals into different frequency bands. 

2. epoching: the continuous EEG signal is cut into trials 

3. CSP computation: per frequency band, the common spatial patterns algorithm is applied to extract spatial filters. CSP aims to extract spatial filters that make the trials discriminable by the power of the spatially filtered trial signal. 

4. spatial filtering: the spatial filters computed in Step 2 are applied to the EEG signal

5. feature construction: feature vectors are the log-variance of the spatially filtered trial signal for each frequency band and for each spatial filter. 

6. classification: A classifier is trained to predict per-trial labels based on the feature vectors. 

Input representation: how to represent input $ X^j $

To represent the EEG as a time series of topographically organized images, i.e., of the voltage distributions across the flattened scalp surface. Unmixing of these global patterns using a number of spatial filters is therefore typically applied to the whole set of relevant electrodes as a basic step in many successful examples of EEG decoding. 

ConvNets are designed in a way that they can learn spatially global unmixing filters in the entrance layers, as well as temporal hierarchies of local and global modulations in the deeper architectures. 

The input will be represented as a 2D-array with **the number of time steps as the width** and **the number of electrodes as the height**. This approach significantly reduced the input dimensionality compared with the "EEG-as-an-image" approach. 

The first convolutional block was split into two convolutional layers in order to better handle the large number of input channels, **one input channel per electrode compared to three input channels (one per color) in rgb-images**.

The convolution was split into a first convolution across time and a second convolution across spcace (electrodes); each filter in these steps has weights for all electrodes (like a CSP spatial filter) and for the filters of the preceding temporal convolution (like any standard intermediate convolutional layer). 

ConvNet architectures and design choice:

Input representation

Deep ConvNet for raw EEG signals

Shallow ConvNet for raw EEG signals

Design choices for deep and shallow ConvNet

Hybrid ConvNet

Residual ConvNet 

ConvNet training:

Input and labels

Trial-wise training 

Cropped training 

Optimization and early stopping 

---

# Pipeline of train_decoders.py

## Purpose: train and fine-tune decoders

1. Data load 

---

2. Tailored decoders parameters (within participant) (_tail)

n_folds: number of folds for each participant

spec_meas: specific measurements? power, power_log, relative_power, phase, freqslide

hyps: hyperparameters of tailored model
F1, dropoutRate, kernLength, kernLength_sep, dropoutType, D, n_estimators, max_depth

epochs: training epochs

patience: ?

---

3. Same modality but across participants decoder parameters

n_folds: number of folds for total folds 

spec_meas: specific measurements? power, power_log, relative_power, phase, freqslide

hyps: hyperparameters of cross-subjects-same-modality decoder

epochs: training epochs

---

4. Unseen modality testing parameters (across participants) 

eeg_roi_proj_lp: projection matrix, **not found**

---

5. Fine-tune parameters for same modality

model_type: select which model? 

layers_to_finetune: which layers to fine-tune or entire model to be retrained 

sp: output path of retrained layers 

train/validation split: use_per_vals, if True use percentage values (otherwise, use number of trials)

percentage of train/val: [0.17， 0.33， 0.5， 0.67], [0.08, 0.17, 0.25, 0.33]

number of trials train/val: [16, 34, 66, 100], [8, 16, 34, 50]

---

6. Train some modality decoders with different numbers of training participants 

**User-defined parameters**: max_train participants, number of validation participants

---

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
  
  kernLength_sep: 
  
  dropoutType:
  
  D:
  
  F2: 
  
  n_estimators:

---

8. Same modality training

run_nn_models()

---

9. unseen modality testing 

unseen_modality_test()

---

10. Same modality fine-tuning

transfer_learn_nn()

```
for j, curr_layer in enumerate(layers_to_finetune)
    # Create save directory if does not exist already
    
    # Fine-tune with each amount of train/val data

```

single_sub: bool, parameter of transfer_learn_nn

lp_finetune: load path

parameters of transfer_learn_nn:

lp_finetune: load path

sp_finetune: path of saving output 

model_type_finetune: 'eegnet_hilb', NN model type to fine-tune, must be either 'eegnet_hilb' or 'eegnet'

layers_to_finetune: specific layers or the entire model

use_per_vals: bool

per_train_trials: train split

per_val_trials: val split 

single_sub: bool, layers_to-finetune

epochs: training epochs

---

11. Unseen modality fine-tuning

**transfer_learn_nn_eeg**

---

12. Training same modality decoders with different numbers of training participants

run_nn_models: 

  sp_curr: rootpath + dataset+ '/single_objs'
  
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
  
  **n_val: number of validation?**
  
  **n_train: number of training**
  
  F1:
  
  dropoutRate:
  
  kernLength:
  
  kernLength_sep: 
  
  dropoutType:
  
  D:
  
  F2: 
  
  n_estimators:
  
---

13. Combine results into dataframes

Two functions: ntrain_combine_df, frac_combine_df

---

14. Pre-compute difference spectrograms for ECoG and EEG datasets 

---

What kinds of ML methods perform well?

HTNet is compare with:

EEGNet

Random forest, (LDA?)

Minimum distance decoders. 

---

Performance of the decoder will be evaluated in three scenarios:

a) testing on **an untrained recording day** for the same ECoG participant (tailored decoder). same data and same subject

b) testing on **an untrained ECoG participant** (same modality). multi-participant decoder for same modality. 

c) testing on **participants on from the EEG dataset after training only on the ECoG dataset** (unseen modality). 

For all scenarios, they performed 36 pseudo-random selections (folds) of the training and validation datasets, such that each of the 12 ECoG participants was the test participant three times. 

The last recording day of each ECoG participant is used as the test set and is excluded from all training and validation sets. 

All training, validation, and test sets were balanced with equal numbers of **move** and **rest** events. 

---
Hyperparameter tuning: 

For HTNet and EEGNet: 6 parameters, [temporal kernel length, separable kernel length, temporal filter count,  dropout rate, dropout type, model type]

For random forest decoder: 2 parameters, [maximum distance, number of estimators]

For minimum distance decoder: no parameters

---

Hyperparameter selection trials: 

Random forest parameter: 25 runs

HTNet/EEGNet: 100 runs 

---
Evaluation: 

Metric: validation accuracy (averaged by 36 folds for same-modality-and-same-subject decoder, 12 folds for multi-participants-same-modality).

---

Fine-tune

The pretrained same and unseen modality decoders are fine-tuned using a portion of the test participant's data. 

In two ways:

1. Each HTNet convolutional layer is fine-tuned separately. The nearby batchnor malization layer is also trained and notably boosted performance. 

2. All layers are fine-tuned together. 

The amount of training/validation data available is varied as well:

17% for training, 8% for validation

33% for training, 17% for validation

50% for training, 25% for validation

67% for training, 33% for validation 

The relationship of test accuracy and logarithm of the number of events used for fine-tuning model is linearly modeled. 

---

2.8 Comparing performance of HTNet spectral measures. 

??? I didn't understand what this is 

---

2.9 Interpreting model weights

dont understand

---

2.10 Effect of training participants on performance 

How many training participants are needed for improving decoder performance?

Assumption: More training participants, better decoder performance. 

Fact: No!

---

Results:

1. Feasibility of transdering to unseen participants. 

---

By fine-tuning thse pretrained HTNet 

Additionally , no significant differences in performance between fine-tuning approaches, but significant improvements in test accracy when comparing fine-tuned decoders trained on the same data. 


1. number of subject: 9

2. number of motor imagery tasks: 4

class 1: movement of left hand, 

class 2: movement of right hand, 

class 3: movement of both feet, 

class 4: tongue 

3. number of sessions: 2, on different days, for each subject

4. number of runs per session: 6

5. number of trials per run: 48 (12 for each of the four classes) 

(48 * 6 = 288 trials per session for each subject) 

---

Experiment scheme: 

1. At the beginning of each session, a recording of approximately 5 mins was performed to estimate the EOG influence. (eyes open, eyes closed, eyes movement) 

The EOG channels are provided for the subsequent application of artifact processing methods and must not be used for classification. 

All data sets are stored in the General Data Format for biomedical signals, **one file per subject and session** (that's why 2 * 9 = 18 files). 

Only one session contains the class labels for all trials, whereas the other session will be used to test the classifier and hence to evaluate the performance. **AXXT** means for training **AXXE** means for testing. 

A GDF file can be loaded as [s, h]

s: signals, the signal variable contains 25 channels (22 EEG + 3 ECOG signals)

h: a header structure h. The header structure contains event information that describes the structure of the data over time. The following fields provide important information for the evaluation of this dataset: The position of an event in samples, the corresponding type, and the duration of that particular event. 

The class labels are only provided for the training data and not for the testing data. (i.e., 1, 2, 3, 4 corresponding to event types 769, 770, 771, 772). The trials containing artifacts as scored by experts are marked as events with the type 1023. 

276 - eyes open

277 - eyes closed

768 - start of a trial 

769 - class 1 (left)

770 - class 2 (right) 

771 - class 3 (foot)

772 - class 4 (tougue) 

783 - unknown

1023 - reject trial

1072 - eye movements

32766 - start of a new run 

---

A continuous classification output for each sample in the form of class labels (1, 2, 3, 4), including trials and trials marked as artifact. 

**A confusion matrix** will be built from all artifact-free trials for **each time point**.

Input: EEG data 

Output: class label vector

It is required to remove EOG artifacts before the subsequent data processing using artifact removal techniques such as highpass filtering or linear regression. 

---

Q1. How to set the number of trials and the number of subjects of one session?

1. number of trials: >= 30, if larger than 40 is better. 

2. number of subjects: >= 8 subjects (for paper). 10 is good.

Q2. The simplest way to get results: 

Find the open source datasets (>=2) and implement CSP. 

Point: It is important to explain the originality of algorithm and data processing. 

Q3. What kinds of work can be further researched? 

1. Improve the spatial resolution of EEG signals. 

  e.g., Combine fMRI data with EEG signals (VBEMG, estimate the spatial location by fMRI data and then infer with EEG signals)
  
2. Estimation of continuous signals. 

3. Inter-subject electrodes location projection. 

4. SLR(Sparse Logistic Regression)

5. CSP

6. Sliding window with CSP(Hilbert transform?) 

7. Design experiments to collect data (points will be the originality of the experiment design) 

8. Time domain, frequency domain features. 

---


