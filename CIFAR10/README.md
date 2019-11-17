# Training CIFAR-10 on ResNet9

We leverage the training of a small ResNet (Resenet9) on CIFAR10 to 94% test accuracy in 79 seconds as described [in this blog series](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/). This folder containts the code for the distributed CIFAR-10 training of ResNet9 and AlexNet on Pytorch using Compressed Gradient Communication. Two approaches for compressing the model gradients are possible. Layerwise approach allows for applying different compression methods on each layer while Entire-model approach treats all the model gradients as a whole and applies compression on them as a single entity.

# Implementation

Details of the compressed communication can be found in `core.py`. `layerwise_compressed_comm` function implements the layerwise version of the compressed gradient communication. `entiremodel_compressed_comm` function implements the enitre-model version of the compressed gradient communication. `run_batches` function implements the training loop and calls upon the proper communication function.

# Supported Models

* Resnet9
* Alexnet

# Supported Methods

* Randomk - K ratio parameter (percent of selected elements)
* TopK - K ratio parameter (percent of selected elements)
* Thresholdv -  V threshold parameter (the value used to pick the elements)
* TernGrad - no parameter
* QSGD -  qstates parameter (number of states of QSGD)

# Compression related script parameters
```
parser.add_argument('--network', '-n', type=str, default='resnet9')
parser.add_argument('compress', '-c', type=str, default='none')
parser.add_argument('--method', type=str, default='none')
parser.add_argument('--ratio', '-K', type=float, default=0.5)
parser.add_argument('--threshold', '-V', type=float, default=0.001)
parser.add_argument('--qstates', '-Q', type=int, default=255)
parser.add_argument('--momentum', type=float, default=0.0)
```

# Running the experiment - an Example
Run the following line on the nodes participating in the training, change the node_rank (rank of each node) and nnodes  (# of nodes) parameters accordingly.

```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE --nnodes=NUM_NODES --node_rank=RANK --master_addr=MASTER_ADDR --master_port=MASTER_PORT training/train_imagenet_nv.py --logdir LOG_DIR --distributed --init-bn0 --no-bn-wd --name RUN_NAME --compress='layerwise' --method='Topk' --ratio=0.3
```


 


