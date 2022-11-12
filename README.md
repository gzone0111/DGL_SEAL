# DGL_STAR_GCN
Rating prediction architecture.
Due to busy workload from university, I only used one dataset(ML-100k) for training and testing, performing inductive rating prediction.
paper url:https://arxiv.org/pdf/1905.13129.pdf
STAR-GCN, STAcked and Reconstructed Graph Convolutional Networks
architecture to learn node representations for boosting the performance in recommender systems, especially in the cold start scenario. 

STARGCN masks a part of or the whole user and item embeddings and reconstructs these masked embeddings with a block of graph encoder-decoder in the training phase.
Use sample-and-remove training strategy because of the label leakage issue, which results in the overfitting problem.

Two task:
transductive rating prediction and inductive rating prediction
![](https://i.imgur.com/LgPJ7ur.png)
1. Transductive rating prediction is traditional way to evaluate model ability
2. Inductive rating prediction is newly introduced task to evaluate model ability according to the paper.

Comparing GC-MC:
**GC-MC limtations**
While being powerful, the GC-MC model has two significant limitations. To distinguish each node, the model uses one-hot vectors as node input. This makes the input dimensionality proportional to the total number of nodes and thus is not scalable to large graphs. Moreover, the model is unable to predict the ratings for new users or items that are not seen in the training phase because we cannot represent unknown nodes as one-hot vectors. The task of predicting ratings for new users or items is also known as the cold start problem.


Architecture:
STARGCN, multiblock graph encoder decoder, each block contains two components: a graph encoder and a decoder. Encoder generate node representation and decoder recover from the representation. (1b2l) will be used
![](https://i.imgur.com/AA7JGvo.png)
