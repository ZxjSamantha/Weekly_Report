## EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

For oscillary-based classficiation of SMR, the traditional approach is the One-Versus-Rest (OVR) filter-bank common spatial pattern. 

---

Classification results are reported for two sets of analysis: within-subject and cross-subject. 

- within-subject classification: a portion of the subjects data to train a model specifically for that subject

- cross-subject classification: use the data from other subjects to train a subject-agnostic model. 
  
  - For the SMR dataset, the data is partitioned as: for each subject, select the training data from 5 other subjects at random to be the training set and the training data from the remaining 3 subjects to be the validation set. The test set remains the same as the original test set for the competition. 
  **Note**: They never use the test subjects' training dat. This process is repeated  10 times for each subject, creating 90 different folds. The mean and standard error of classification performance were calculated over the 90 folds. 
  
- class-weight to the loss function whenever the data is imbalanced. 
