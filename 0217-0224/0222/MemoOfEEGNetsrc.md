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
