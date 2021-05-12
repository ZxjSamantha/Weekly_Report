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





