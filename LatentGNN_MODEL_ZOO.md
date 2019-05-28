## LatentGNN Model Zoo and Baselines

### Hardware
- 4 NVIDIA P40 GPUs

### Software
- PyTorch version: 1.1.0
- CUDA 9.0
- CUDNN 7.5.1
- NCCL 2.2.13-1

### End-to-end Mask R-CNN(+LatentGNN) baselines

All the baselines were trained using the exact same experimental setup as in maskrcnn-benchmark.
We initialize the detection models with ImageNet weights from Caffe2, the same as used by maskrcnn-benchmark.

The pre-trained models are available in the link in the model id.

backbone | type | stage| lr sched | im / gpu | box AP | mask AP | model id
 --  | --   | --   | --       |       -- |     -- |      -- | -- 
R-50-FPN | Mask | - | 1x | 2 | 37.8 | 34.2 | [6358792](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth)
R-50-FPN-LatentGNN | Mask | C3 | 1x | 2 | 38.2 | 34.7 | []()
R-50-FPN-LatentGNN | Mask | C4 | 1x | 2 | 39.0 | 35.2 | [Done]()
R-50-FPN-LatentGNN | Mask | C5 | 1x | 2 | 38.8 | 35.0 | []()
R-50-FPN-LatentGNN | Mask | C345 | 1x | 2 | 39.5 | 35.6 | []()
R-101-FPN | Mask | - | 1x | 2  | 40.1 | 36.1 | 
R-101-FPN-LatentGNN | Mask | C4   | 1x | 2 | 41.0 | 36.9 | 
R-101-FPN-LatentGNN | Mask | C345 | 1x | 2 | 41.4 | 37.2 | [Done]()
X-101-32x8d-FPN     | Mask | -    | 1x  | 1 |42.2 | 37.8 | 
X-101-32x8d-FPN-LatentGNN | Mask | C4   | 1x |  1 | 43.0 | 38.5 | 
X-101-32x8d-FPN-LatentGNN | Mask | C345 | 1x |  1 | 43.2 | 38.8 | 

### Training speed

### Experimenal Records
- R-50-A3
- R-50-A4
  - P40,e2e_mask_rcnn_R_50_FPN_A4_1x_kernel1_exp2
- R-50-A5
- R-50-A345
  - P40, e2e_mask_rcnn_R_50_FPN_A345_1x_kernel1_exp1
- R-101-A4
- R-101-A345
  - P40, e2e_mask_rcnn_R_101_FPN_A345_1x_kernel1_exp4
- X-101-A4
- X-101-A345