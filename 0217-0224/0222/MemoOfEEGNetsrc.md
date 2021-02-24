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

