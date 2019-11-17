# Training Imagenet on ResNet50

This Folder containts the code for the distributed ImageNet training of ResNet50 on Pytorch using Compressed Gradient Communication. Two approaches for compressing the model gradients are possible. Layerwise approach allows for applying different compression methods on each layer while Entire-model approach treats all the model gradients as a whole and applies compression on them as a single entity.

# Implementation

Details of the compressed communication can be found in `train_imagenet_nv.py`. `layerwise_compressed_comm` function implements the layerwise version of the compressed gradient communication. `entiremodel_compressed_comm` function implements the enitre-model version of the compressed gradient communication. `train` function implements the training loop and calls upon the proper communication function.

# Supported Methods

* Randomk - K ratio parameter (percent of selected elements)
* TopK - K ratio parameter (percent of selected elements)
* Thresholdv -  V threshold parameter (the value used to pick the elements)
* TernGrad - no parameter
* QSGD -  qstates parameter (number of states of QSGD)

# Compression related script parameters
```
parser.add_argument('compress', '-c', type=str, default='none')
parser.add_argument('--method', type=str, default='none')
parser.add_argument('--ratio', '-K', type=float, default=0.5)
parser.add_argument('--threshold', '-V', type=float, default=0.001)
parser.add_argument('--qstates', '-Q', type=int, default=255)
parser.add_argument('--momentum', type=float, default=0.0)
```

# Running the experiment - an Example 
Run the following line on the nodes participating in the training, change the rank (rank of each node) and world_size  (# of nodes) parameters accordingly.

```
python dawn.py --master_address='tcp://192.168.0.1:2222' --rank=0 --world_size=1 --network='resnet9' --compress='layerwise' --method='Topk' --ratio=0.3
```