## BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data 

**Problem**: 

DNNs are used to learn general features and these features could be fine-tuned to specific contexts. 

In other words, unlike domains such as computer vision where there is clearer understanding that nearly all DNNs tend to learn "low-level" features in earlier laters (e.g., edge-detector-like primitives), there is no such understanding with DNNs used to process raw EEG. There are no known transferable DNN properties or operations that are easily extended to any subject, session, or task. 

**Insights**: 

With these shallower networks, the range of learnable features is relatively limited. Their inability to uniformly outperform feature-engineering approaches indicate that these limited features are not entirely sufficient, and more importantly, they may not always be desirable in a DNN approach. 

More complex raw-BCI-trial features could be developed using DNNs with sufficient data, notably such that these data provide a reasonable empirical estimate of the data distribution in question. 

**Important question**:

How best to construct such an EM, so that it learns features that are general enough while remaining usable for any analysis task? VS **Constrastive learning**

## Inter-subject Deep Transfer Learning for Motor Imagery EEG Decoding 

(标题几乎一模一样woc）

### Objectives: Address **Negative transfer**, i.e., CNNs learning from dissimilar EEG distributions from different subjects causes CNNs to misrepresent each of them instead of learning a richer representation. 

They proposed a multi-branch deep transfer network, the Separate-Common-Separate Network (SCSN) based on splitting the network's feature extractors for individual subjects. 

### Results: SCSN(81.8%m 53.2%) and SCSN-MMD(81.8%, 54.8%), CNN(73.4%, 48.8%). 

2 strategies:

1. Fine-tuning 

2. To train a shared network using multiple datasets but split deep network layers for different datasets. 

### Main Idea

To address negative transfer problem: 

They design a Separate-Common-Separate Network(SCSN) by separating feature extractor of the baseline CNN for individuals, so **each of the subjects has their own temporal layer, spatial layer and mean pooling layer**.(Feature extraction for each subject?) Each network branch extracts subject-specific features, which could avoid negative transfer in the common feature extractor of the baseline CNN. 

Maximum-mean discrepancy (MMD) is a metric which meaures the distance between two datasets in kernel sapce. They compute MMD between the target and each source subject in the separate deep feature extractors and add it into the loss function. 

They compute MMD for each of the three layers across subjects. The $(MMD_i)^2$ loss is a weighted average MMD of each of the three layers. Their averaging weights are 1/6, 1/3, 1/2 respectively; this is to increase the significance of deeper layers. 

They also match samples with the same label when they compute MMD. This ensures the MMD loss represents the distributional distance between data with the same label from different subjects. 

### Data processing pipeline 

Five of the nine subjects with highest data quality (subject 01, 03, 07, 08, 09) are selected from the dataset. 

**In my case, subejct 01, 03, 05, 07, 08**

1. They perform a 50 Hz notch filter and a band pass filter between `[1-100]` Hz.

2. They then crop each trial into 2-second trials with an overlap of 1.9 seconds to better fit the real-time setup. 

3. Data split: 

For the BCICIV2a dataset, they simulate the calibration period in real-world BCI by including **the target subject's first 120 trials of the second session into the training set**. 

**The training set consists of the first session of all five subejcts as well as the target subject's first 120 trials of the second session. **

**The validation set contains the  `[120, 144]` trials of the target subject's second session.** 

**The last 144 trials in the second session form the test set. **

Batch size: 30 

Finally, comparing SCSN and SCSN-MMD, they did not observe an obvious increase in decoding accuracies by adding MMD constraints to our multi-subject networks. 

---

## Chap 8

预训练方法：首先在大数据集上训练得到一个具有强泛化能力（how to specify this）的模型（预训练模型），然后在下游任务上进行微调的过程。

深度迁移学习方法：在预训练的基础上，设计更好的网络结构，损失函数等，从而更好地迁移。


一种被广泛接受的解释如下：对于神经网络而言，其浅层负责学习general features, 其深层则负责学习与任务相关的specific features. 随着层次的加深，网络渐渐从通用特征过渡到特殊特征的学习表征。这意味着，如果能准确地指导一个网络中哪些层负责学习通用的特征，哪些层负责学习特殊的特征，那么就能更清晰地利用这些层来进行迁移学习。

---

They consider how to adapt techniques and architectures used for **language modeling(LM)**, that appear capable of ingesting awesome amounts of data, towards the development of encephalography modeling (EM) with DNNs in the same vein. 

[GNN Introduction](https://distill.pub/2021/gnn-intro/)

---

## Adapting Visual Category Models to New Domains

### Experimental Protocols

- Same-category setting:
- New-category setting: 

## Deep Domain Confusion: Maximizing for Domain Invariance 

Optimizing for domain invariance can be considered equivalent to the task of learning to predict the class labels while simultaneously finding a representation that makes the domains appear as similar as possible. They learn deep representations by optimizing over a loss which includes both classification error on the labeled data as well as a domain confusion loss which seeks to make the domains indistinguishable. 

Our intuition is that if we can learn a representation that minimizes the distance between the source and target distributions, then we can train a classifier on the source labeled data and directly apply it to the target domain with minimal loss in accuracy. 

To minimize the loss, one approach is to take a fixed CNN, which is already a strong classification representation, and use MMD to decide which layer to use activations from to minimize the domain distribution distance. 

1. Take a network that was trained to minimize $lambda_C$ 

2. Select the representation that minimizes MMD 

3. Use that representation to minimize $lambda_C$. 

*bottleneck* -> adaptation layer 

Our intutition is that a lower dimensional layer can be used to regularize the training of the source classifier and prevent overfitting to the particular nuances of the source distribution. 

There are two model selection choices that must be made to add the adaptation layer and the domain distance loss. We must choose *where in the network to place the adaptation layer*(solved by using MMD) and *choose the dimension of the layer*(solved by grid search) 

Both the selection of *which layers's representation to use(depth)* and *how large the adaptation layer should be(width)* are guided by MMD. **what do these representation mean in eeg data?**


## Parameter Transfer Unit for Deep Neural Networks

Existing works usually heuristically apply parameter-sharing or fine-tuning, and there is no principled approach to learn a parameter transfer strategy. Two popular parameter-based transfer learning methods are *parameter-sharing* and *fine-tuning*. Parameter-sharing assumes that the parameters are highly transferable, and it directly copies the parameters in the source domain network to the target domain network. The fine-tuning method assumes that the parameters in the source domain network ar useful, but they need to be trained with target domain data to better adapt to the target domain. 

### Limitations of parameter-based transfer learning

1. The parameter transferability is mannualy defined as discrete states, usually "random", "fine-tune" and "frozen". But the transferability at a fine-grained scale has not been considered.

2. The parameter transferability differs with domains and network architectures. The hold-out method is rather inefficient because it involves long training time and tremendous computational costs. 

## TACNet: Task-aware EEG Classification for Brain-Computer Interface through A Novel Temporal Attention Convolutional Network

Why temporal filter in CNN: However, assuming all input contains the same amount of information, most CNN models apply the same filter structure and parameters uniformly across the input. Therefore, enabling the CNN model to selectively focus on signal slices that contain more task-related information is a feasible solution to the problem of EEG non-stationarity. 

## Convolutional Neural Network-based Transfer Learning and Knowledge Distillation using Multi-Subject Data in Motor Imagery BCI

Two main approaches: domain adaptation, rule adaptation 

They take a RA-based approach to the problem of classifying motor imagery EEG signals: rather than bringing the data to a common space, they train a neural network model that captures information from multiple subjects and stores the information as the parameters of the network. 

### EEG Representations

Using FBCSP algorithm, they first find the **spatial filters** and **frequency bands** that are contributing to the discriminance between the classes based on the **log-energy** features. 

The results show a significant increase in both small subset calibration in the three cases shown. 

## Deep Transfer Learning for EEG-based Brain-Computer Interface 

### Problems:

1. Traditional methods do not fully exploit multimodal information. 

2. Large-scale annotated EEG datasets are almost impossible to acquire because biological data acquisition is challenging and quality annotation is costly. 

### Workflow

They convert raw EEG signals into EEG optical flow to represent the multimodal information. 

不太一样

## Convolutional Neural Network-based Transfer Learning and Knowledge Distillation using Multi-subject Data in Motor Imagery BCI

### Workflow

- Extract EEG representations from multiple subjects independently. 

- Uses a deep convolutional neural network to train a model on the multi-subject data

- Transfers the model parameters to train/fine-tune on the new subject's data 

- Utilizes the labels estimated by the transferred model to regularize the training/fine-tuning process. 

**EEG representation**

Input(features): The envelop power, extracted using absolute value of the analytic signal. (FBCSP) 



---

`
Conventional machine learning algorithms are composed of three steps: signal pre-processing, feature extraction, feature classification. 

The signal pre-processing step's objective is to remove artifacts such as musclocular movement and system noises. 

For feature classification, popular linear classifiers such as SVM and LDA are applied for MI classification. Most popular classification algorithms are linear, which is not suitable for non-stationary signal classification. 
`

---

`基于模型的迁移学习方法，即构建参数共享的模型。例如，SVM的权重参数、神经网络的参数等，都可以进行共享。由于神经网络的结构可以直接进行迁移，因此其使用频率非常高。神经网络最经典的pretrain-finetune就是模型参数迁移的很好的表现。`

---

## MMD

[Implementation of MMD](https://blog.csdn.net/sinat_34173979/article/details/105876584)

---

## Transfer Learning for EEG-based Brain-Computer Interfaces: A review of progress made since 2016 

`Usually, a calibration session is needed to collect some training data for a new subject, which is time-consuming and user unfriendly. Transfer learning, which utilizes data or knowledge from similar or relevant subjects/sessions/devices/tasks to facilitate learning for a new subject/session/device/task, is frequently used to reduce the amount of calibration effort. `

`EEG signals are weak, easily contaminated by interference and noise, non-stationary for the same subject, and varying across different subjects and sessions. Therefore, it is challenging to build a universal machine learning model in an EEG-based BCI system that is optimal for different subjects. Reducing subject-specific calibration is critical to the market success of EEG-based BCIs.`

TL scenarios:

`Cross-subject TL: Data from other subjects are used to facilitate the calibration for a new subject(the target domain). Usually, the task and EEG device are the same across subjects.`

Formulate the TL on BCI Competition datasets:

TL in deep learning: 

`Currently, a common TL technique for deep learning-based EEG classification is based on fine-tuning with new data from the target session/subject. Unlike concatenating target data with existing source data, the fine-tuning process is established on a pre-trained model and performs iterative learning on a relatively small amount of target data. `

---
## Deep mapping with CNN for brain mapping and decoding of movement-related information from the human EEG

`Another findings of the study was that shallow ConvNets performed as good as the deep ConvNets, in contrast to the hybrid and residual architectures. These observations could possibly be better understood by investigating more closely what discriminative features there are in the EEG data and what architectures can hence best use them. ` 

feature和architecture 的对应关系

`It would be interesting to study the effect of more layers when the networks use mostly EEG band power features, phase-related features, or a combination thereof and whether there are features for which a deeper hierarchical representation could be beneficial. `

`ConvNets did not improve accuracies over FBCSP by a large margin. Significant improvements, if present, were never larger than 3.5 percent on the combined dataset with a lot of variation per subject. However, the deep ConvNets as used here may have learned features different from FBCSP, which could explain their higher accuracies in the lower frequencies where band power features may be less important. `

deepConvNets学到的features和FBCSP不一样（可视化每层学到的特征！！！）（分frequency band同步学习？）

The possible reasons that ConvNets failed to clearly outperform FBCSP:

### Potential advantages of ConvNets for brain-signal decoding

```
Besides the decoding performance, there are also other potential advantages of using deep ConvNets for brain-signal decoding. Several use cases desirable for brain-signal decoding are very easy to do with deep ConvNets iteratively trained in an end-to-end fashion: 

1. workload estimation, error- or event-related potential decoding. 

2. Due to the iterative training of ConvNets, they have a natural way of pretraining and finetuning; for example a ConvNet can be pretrained on data from the past or data from other subjects and then be finetuned with new data from a new subject. 


```

### Cropped training effect on accuracies

`Cropped training was necessary for the deep ConvNet to reach competitive accuracies on the dataset excluding very low frequencies. The large increase in accuracy with cropped training for the deep network on the data might indicate a large number of training examples is necessary to learn to extract band power features. This makes sense as the shifted neighboring windows may contain the same, but shifted, oscillatory signals. These shifts could prevent the network from overfitting on phase information within the trial, which is less important in the higher than the lower frequencies. `


