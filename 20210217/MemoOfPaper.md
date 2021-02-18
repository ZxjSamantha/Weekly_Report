# Subject-Independent Brain-Computer Intefaces Based on Deep Convolutional Neural Networks

It is an interesting topic to build a calibration-free or subject-independent BCI. 

The database is composed of 54 subjects performing the left- and right-hand MI on two different days, resulting in 21600 trials for the MI tasks. 
(?why the type of tasks are so limited?) 

## Discussion 

It is interesting to investigate how many data are truly needed to develop an acceptable subject-independent model with a DL framework. (i.e., what is the number of training samples required to train the CNN). 

A small number of training samples could either help or harm the actutal training. (considering negative transfer) 

A large-scale MI database and a proper configuration of the filter bank is also an essential factor in influencing the decoding accuracy of the subject-independet BCI based on DL. 

Visualization of the proposed input and the 28 by 28 activations on each convolutional layer. 

## Limitations and Future Study

The proposed method is based on the CNN structure and there are several new architectures that are worthwhile to explore. 
