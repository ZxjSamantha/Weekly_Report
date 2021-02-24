# Implementation of FBCSP

As in many previous studies, we used regularized linear discriminant analysis as the classifier, with shrinkage regularization. 

To decode multiple classes, we used one-vs-one majority weighted voting: we trained an RLDA classifier for each pair of classes, summed the classifier outputs across classes and picked the class with the highest sum.
