
1. number of subject: 9

2. number of motor imagery tasks: 4

class 1: movement of left hand, 

class 2: movement of right hand, 

class 3: movement of both feet, 

class 4: tongue 

3. number of sessions: 2, on different days, for each subject

4. number of runs per session: 6

5. number of trials per run: 48 (**12** for each of the four classes) 

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










