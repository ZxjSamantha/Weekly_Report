## Contributions: 5 schemes for adaption of a deep CNN with limited EEG data. 84.19% (9.98%). 

### The effect of different learning rates and percentages of adaptation data. 

## Baseline: Schirrmeister(2017) 

### 2 baselines to compare the performance of the proposed subject adaption: subject-specific and subejct-independent classification. 

- Subject-specific classfication: Train and validate a model for each subject using only the same subject's data. 

- Subject-independent classification: **Leave-one-subejct-out(LOSO)** paradigm for evaluation. 

## Goal: Leverage the features extracted from the convolution filters in the model and adapt the classifier to a subject it has never encountered. 

### Classification accuracy for motor imagery of SOTA approaches: 60% - 80% 

### Previous Transfer Learning Methods:

1. FBCSP 

2. Spectral-spatial input generation

3. Online pre-alignment strategy based on **Riemannian Procrustes Analysis** （RPA), TL benefits for both DL and ML.  

4. **Mainfold Embedded Knowledge Transfer (MEKT)** with a combination of alignment, feature extraction, and domain adaptation techniques to produce projection matrices that minimize the joint probability distribution shift between the source and the target domains. 

5. Number of features?

6. Different learning rates can have an impact on the final classification accuracy. 

Learning rates & Scheme: 


---

Transformer on MI?

