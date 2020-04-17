### VSGNet
Code for our CVPR 2020 paper [**VSGNet:Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions.**](https://arxiv.org/abs/2003.05541) 

## Citing
If you find this work useful, please consider our paper:

	@article{ulutan2020vsgnet,
  		title={VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions},
  		author={Ulutan, Oytun and Iftekhar, ASM and Manjunath, BS},
  		journal={arXiv preprint arXiv:2003.05541},
  		year={2020}
	}

## Installation
1. Clone repository (recursively):
```Shell
git clone --recursive https://github.com/ASMIftekhar/VSGNet.git
```
2. Download data,annotations,object detection results:
```Shell
bash download_data.sh
```
You need to have wget and unzip packages to execute this script. Alternatively you can download these files from [here].(https://drive.google.com/drive/folders/1J8mN63bNIrTdBQzq7Lpjp4qxMXgYI-yF?usp=sharing)
If you execute the script then there will be two folders in the directory "All\_data\_vcoco" and "infos". This will take close to 1.9GB space. Inside the All\_data\_vcoco folder you will find the following subdirectories.

**a.Data_vcoco**: It will contain all training and validation images of v-coco inside train2014 subdirectory and all test images of v-coco inside val2014 subdirectory.

**b.Annotations\_vcoco**: It will contain all annotations of training, validation and testing set in three json files. The annotation is taken from v-coco API and converted into our convenient format. For example, lets consider there is only one single image annotated with two verbs "smile" and "hold" along with two persons and object bounding boxes. The annotation file will be arranged as follows:

```
	{image_id:[{'Verbs': 'hold',
  	'object': {'obj_bbx': [305.84, 59.12, 362.34, 205.22]},
  	'person_bbx': [0.0, 0.63, 441.03, 368.86]},
 	{'Verbs': 'smile',
  	'object': {'obj_bbx': []},
  	person_bbx': [0.0, 0.63, 441.03, 368.86]}]}
```
**c.Object\_Detections\_vcoco**: It will contain all object detection results for v-coco. 

**d.v-coco**: It will contain original v-coco API. This is needed for doing evaluations.







The code will be released within April 2020. 

Please contact A S M Iftekhar (iftekhar@ucsb.edu) for any queries.
