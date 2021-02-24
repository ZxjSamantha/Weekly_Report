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

