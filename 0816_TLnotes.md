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

