## Data processing pipeline (Inter-subject Training)

Five of the nine subjects with **highest data quality (subject 01, 03, 07, 08, 09)** are selected from the dataset. 

**In my case, subejct 01, 03, 05, 07, 08**

1. They perform a 50 Hz notch filter and a band pass filter between `[1-100]` Hz.

2. They then crop each trial into 2-second trials with an overlap of 1.9 seconds to better fit the real-time setup. 

3. Data split: 

For the BCICIV2a dataset, they simulate the calibration period in real-world BCI by including **the target subject's first 120 trials of the second session into the training set**. 

**The training set consists of the first session of all five subejcts as well as the target subject's first 120 trials of the second session. **

**The validation set contains the  `[120, 144]` trials of the target subject's second session.** 

**The last 144 trials in the second session form the test set. **

Batch size: 30 

## Cross-subject data partition

For the SMR dataset, the data is partitioned as follows: For each subject, 

select the **training data** from 5 other subjects at random to be the **training set**, 

and the **training data** from the remaining 3 subjects to be the **validation set**.  

The **test set** remains the same as the original **test set** for the competition. 

This enforces a fully cross-subject classification analysis as we never use the test subjects' training data. 

This process is repeated 10 times for each subject, creating 90 different folds. The mean and standard error of classification performance were calculated over the 90 folds. 

## EEG-Adapt

subjetc-specific

subject-independent 

subject-adaptive: **Pretrain-Finetune**. In subject-adaptive classification, they fine-tune and adapt a pre-trained model using a small amount of data from the target subject. 

For each target subject, the model trained in subject-independent classification serves as a pre-trained model. 

**In subject-adaptive classification, they fine-tune and adapt a pre-trained model using a small amount of data from the target subject. **

Scheme 1: adapt the fully-connected layer, leaving the parameters for the feature extractor unchanged. 

**Learning rate scheduling in adaptation**

In these schemes, since the adaptation data is small comparing to the data to train the subject-independent model, we need to tune down the learning rate to avoid cloberring the initialization. 

Original learning rate in the base model be $/eta$, and let $/alpha$ be the coefficient that scales down the learning rate. 

The equation above showed that scaling down the learning rate can be viewed as accepting only $/alpha$ portion of the new parameters. 

In the following sections, an adaptation scheme is defined as optimizing a subset of $/theta$ for f. The adaptation rate is defined as the percentage of available adaptation data used in each scheme, which ranges from 10% to 100% in steps of 10%. An adaptation configuration is a combination of an adaptation scheme. 



## Chap 6 统计特征变换迁移法（就是MMD啦！）

Maximum Mean Discrepancy: 求两个概率分布映射到另一个空间中的数据的均值之差

给定不同的核函数，就可以算出不同的MMD -> Mutiple-Kernel MMD 

基于MMD 进行迁移学习方法的步骤如下：$$A^TXMX^TA$$

1. 输入两个特征矩阵，首先用一个初始的简单分类器（如KNN）计算目标域的伪标签。

2. 随后计算M和H矩阵，然后选择一些常用的核函数进行映射计算K，接着求解公式中的A，取其前m个特征值。

3. 得到源域和目标域降维后的数据，可多次迭代。

### 度量学习法

在特定任务中单纯地运用显式度量达不到预期的效果，我们可以通过在度量方面的研究更好地学习这些距离。此时，这个距离是隐式的。

度量学习的基本思路是：

1. 给定一些训练样本，这些样本中包含了我们预先观测到的一些对样本的知识（先验）（例如，哪两个样本的距离应该要更近一些，哪两个要更远一些）。

2. 建立学习算法，以先验知识为约束条件，构建目标函数，学习到这些样本之间的一个很好的度量。-> 一种特定条件下的优化问题。




## References

[SCNN](https://github.com/SPOClab-ca/SCNN)

[ECE](https://github.com/krishk97/ECE-C247-EEG-GAN)

Preprocessing 快解决了

Training 封装好了

## Hyperparameters + preprocessing 

[cheatsheet and tricks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)

[PyTorch 损失函数](https://lhyxx.top/2020/02/08/Pytorch%E5%B8%B8%E7%94%A8%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E6%8B%86%E8%A7%A3/)

- Data processing 
  - Data augmentation
  - Batch normalization
  - 

- Training a neural network 
- Finding optimal weights
- Parameter tuning 
  - weights initilization 
    - Xavier initialization (Kaiming initialization) 
    - Transfer Learning 
      Small: Freeze all layers, trains weights on softmax
      Medium: Freeze most layers, trains weights on last layers and softmax  

- Optimizing convergence 
  - Learning rate 
  - Adaptive learning rates 

- Regularization 
  - Dropout
  - Weight regularization 
  - Early stopping: This regularization techniques stops the training process as soon as the validation loss reaches a plateau or starts to increase. 

### Hyperparameters:

1. model 

2. batchsize

3. n_epochs

4. src 

5. tar 

6. n_class

7. learning rate 

8. momentum 

9. decay 

10. early stop 

## DaNN Experiment Framework 

1. They first compared the DaNN to baselines and other recent domain adaptation methods. 

2. They investigated the effect of the MMD regularization by measuring the difference of the first hidden layer activations between one domain to another domain. 

The DaNN model used in the expriments has only one hidden layer, i.e., a shallow network of 256 hidden nodes. 

Input: raw pixels, SURF features

Output: ten nodes corresponding to the ten classes. 

The performance of our model was then compared to SVM-based baselines, two existing domain adaptation methods, and a simple neural network as follows: 

1. L-SVM

2. L-SVM + PCA

3. GFK

4. TSC 

5. NN 

### In-domain Setting

In-domain setting, where the training and test samples come from the same domain. The in-domain performance can be considered as a reference that indicates the effectiveness of domain adaptation models in dealing with the domain mismatch. -> 10-fold cross-validation 



## Chap 4 迁移学习方法总览

### 4.1 迁移学习总体思路

迁移学习的核心是找到源域和目标域之间的相似性，并加以合理利用。

度量工作的两个目标：

1. 度量两个领域的相似性，定性地说明它们是否相似，定量地给出相似程度。

2. 以度量为准则，增大两个领域之间的相似性，完成迁移学习。

迁移学习要解决的是**源域和目标域的联合分布差异度量**。 

由于结构风险最小化的准则在机器学习中非常通用，迁移学习可以被统一表征为：

迁移学习的问题可以大体被概括为寻找合适的迁移正则化项的问题。相比传统的机器学习，迁移学习更强调发现和利用**源域和目标域之间的关系**，并将此表征作为学习目标中最重要的一项。 

#### 样本权重迁移法

#### 特征变换迁移法

#### 模型预训练迁移法



---

## Chap 9

在实际应用中，共享层和迁移层采取不同的学习步长，或者直接冻结共享层。

#### 数据分布自适应

1. Marginal Distribution Adaptation

分类器前一层加入自适应度量（Deep Domain Confusion)。

2. Conditional, Joint and Dynamic Adaptation 

Deep Subdomain Adaptation Network: 通过**基于概率**进行类别匹配的软标签机制展开深度迁移

Joint Adaptation Network: 利用多层网络的张量积，定义了联合概率分布在RKHS（?）中的嵌入表达。

Deep Dynamic Adaptation Network: 

DDAN， DDC， JAN等方法均采取了相同的网络结构信息，通过在**特征层**嵌入动态适配单元。

#### 结构自适应

1. Batch Normalization in Transfer Learning: Adaptive Batch Normalization AdaBN:

先在源域书局上用BN操作，然后在新的领域数据如目标域上，将网络的BN统计量重新计算一遍。

Similar: Automatic Domain Alignment Layers

2. 基于多表示学习的迁移网络结构 （Muti-representation Adaptation Network (MRAN))

大多数领域自适应方法使用但一结构将两个领域的数据提取到同一个特征空间，在该特征空间下使用不同方式衡量两个领域分布的差异，最小化这个分布的差异实现分布对齐。

**但是但一结构提取的特征表示通常只能包含部分信息**,所以只在单一结构提取的特征上作特征对齐也只能关注到部分信息，为了更全面地表示原始数据，需要提取多种表示。


#### 知识蒸馏 （Knowledge Distillation)

教师网络 -> 学生网络 

Guan组已经有相应的work了

[Knowledge Distillation on BCI](https://ieeexplore.ieee.org/abstract/document/8008420)



---
IV-2a的时间窗口的长度是怎么设计的？

```
time_windows_flt = np.array([[2.5, 3.5], [3,4], [3.5,4.5], [4,5], [4.5,5.5], [5,6], [2.5,4.5], [3,5], [3.5,5.5], [4,6], [2.5,6]])*self.fs
# time windows in [s] * fs for using as a feature 
self.time_windows = time_windows_flt.astype(int)
```

---

### Chap 9 

深度网络的与训练-微调（pretrain-finetune) 可以节省训练时间，提高学习精度。但是预训练方法有其先天不足：它无法直接处理训练数据和测试数据分布不同的情况，并且，微调时需要有数据标注。

深度迁移学习的核心问题是研究深度网络的可迁移性，以及如何利用深度网络来完成迁移任务。

以数据分布自适应方法为参考，许多深度学习方法都开发出了自适应层(Adaption Layer)来完成源域和目标域数据的自适应。


---

DaNN -> DDC -> DAN 

---

### Chap 9 深度迁移学习的网络结构

最基础的结构的输入数据只包含一个来源，如果不添加新的约束，网络本身无法得知输入数据来自源域还是目标域

1. 单流结构

e.g., 知识蒸馏 knowledge distillation

**知识蒸馏**：知识蒸馏设计了一种teacher-student网络来进行知识的迁移，其核心观点是，一个训练好的复杂模型（教师网络）中蕴含的知识可以被“蒸馏”到另一个小模型中。小模型可以拥有比大模型更简单的网络结构，同时其预测效果也与大模型相近。

2. 双流结构

对于前L层，可以选择共享部分层、微调部分层。**需要探索！！！**

绝大多数工作的不同之处在于**迁移正则项**的设计，这也是不同深度迁移方法的本质区别。

**瓶颈层**：为什么需要瓶颈层？瓶颈层通常是一层网络，其神经元个数少于其接受输入的层，因此瓶颈层往往能获得更为紧致的特征表达，大大提高训练速度。

**Q1：源域的大小和目标域的大小怎么平衡？** 好像不需要size平衡？

Trick 1: 在分类器前一层加入自适应可以达到最好的效果


---

### Chap 8 如何让深度网络自己学会在何时、何处迁移

- 预训练模型可以获得大量任务的通用表现特征general features， 能否直接将预训练模型作为特征提取器，从新任务中提取特征， 从而进行后续的迁移学习。-> 特征嵌入 + 模型构建

- 预训练模型的应用方法：

  - 预训练网络直接应用于任务

  - 预训练+微调

  - 预训练网络作为新任务的特征提取器

  - 预训练提取特征加分类器构建

---

## Feature-based transfer & Model-based transfer


---

### Feature extraction:

Time domain

Frequency domain 

Time-frequency domain 

Riemannian space

functional brain connectivity features 

---

## 未完成

[Cropped Mannual Training](https://robintibor.github.io/braindecode/notebooks/Cropped_Manual_Training_Loop.html)
---

1. 调整卷积窗口大小

2. 调整number of output channels

3. number of convolution layer 

4. number of fully-connected layer 

5. learning rate, weights initialization, number of epochs. 

---

## minimum number of subjects to predict an unseen subject?

## Ensemble learning??? Spectral Transfer learning using Geometry Information

## One large dataset for training the network and one small dataset for fine-tune?  

## compare with other transfer learning methods? compare with non-transfer learning methods?

## with or without pre-training? 

## Small-sample learning? Zero-shot learning? 

## Computational Motor Imagery?

## IV-2a methods on GigaDB dataset? GigaDB methods on IV-2a dataset?

## Source code reading of EEGNet

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


