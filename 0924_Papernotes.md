## Parallel Convolutional-Linear Neural Network for Motor Imagery Classification 



---

`Conventional machine learning algorithms are composed of three steps: signal pre-processing, feature extraction, feature classification. 

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

1. The datasets might still not be large enough to reveal the full potential of deeper convolutional networks in EEG decoding. -> **How to address: increase the size or use transfer learning**

2. The class-discriminative features might not have enough hierarchical structure which deeper ConvNets could exploit. 

Solutions:

1. Recurent networks could exploit signal changes that happen on longer timescales, e.g., **electrodes slowly losing scalp contact over course of a session**, changes of the electrode cap position or nonstationarities in the brain signals.

2. Advances in DL: newer forms of hyperparameter optimization. 

### Potential advantages of ConvNets for brain-signal decoding

```
Besides the decoding performance, there are also other potential advantages of using deep ConvNets for brain-signal decoding. Several use cases desirable for brain-signal decoding are very easy to do with deep ConvNets iteratively trained in an end-to-end fashion: 

1. workload estimation, error- or event-related potential decoding. 

2. Due to the iterative training of ConvNets, they have a natural way of pretraining and finetuning; for example a ConvNet can be pretrained on data from the past or data from other subjects and then be finetuned with new data from a new subject. 

3. Due to their joint optimization, single ConvNets can be building blocks for more sophisticated setups of multiple ConvNets. 

```

training -> cross training strategies -> finetune (finetune strategies) 

### Cropped training effect on accuracies

`Cropped training was necessary for the deep ConvNet to reach competitive accuracies on the dataset excluding very low frequencies. The large increase in accuracy with cropped training for the deep network on the data might indicate a large number of training examples is necessary to learn to extract band power features. This makes sense as the shifted neighboring windows may contain the same, but shifted, oscillatory signals. These shifts could prevent the network from overfitting on phase information within the trial, which is less important in the higher than the lower frequencies. `

### Visualize what ConvNets learn from the EEG data. 

`The literature on using ConvNets for brain-signal decoding has visualized weights or outputs of ConvNet layers determined inputs that maximally activate specific convolutional filters. `


