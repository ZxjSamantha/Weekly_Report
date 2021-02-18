# Graph Neural Network
[A Gentle Introduction to Graph Neural Networks](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)

Graph Neural Network is a types of neural network which directly operates on the Graph structure. A typical application of GNN is node classification. 

Essentially, every node in the graph is associated with a label, and we want to predict the lable of the nodes without ground-truth. 

In the node classification problem setup, each node v is characterized by its feature x_v and associated with a ground-truth label t_v. 

Given a partially labelled graph G, the goal is to leverage these labeled nodes to predict the labels of the unlabeled. It learns to represent each node with a d dimensioanl vector (state) h_v which contains the information of its neighborhood. 

DeepWalk is the first algorithm proposing node embedding learned in an unsupervised manner. It highly resembles word embedding in terms of the training process. 
