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

All training, validation, and test sets were balanced with equal numbers of move and rest events. 

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

Each HTNet convolutional layer is fine-tuned separately. The nearby batchnor malization layer is also trained and notably boosted performance. 

All layers are fine-tuned together. 

The amount of training/validation data available is varied as well.

---

2.8 Comparing performance of HTNet spectral measures. 

4 re-trained models per fold. (36 or 12) 
