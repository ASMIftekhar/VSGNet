### VSGNet
Code for our CVPR 2020 paper [**VSGNet:Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions.**](https://arxiv.org/abs/2003.05541) 

## Citing
If you find this work useful, please consider our paper to cite:

	@article{ulutan2020vsgnet,
  		title={VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions},
  		author={Ulutan, Oytun and Iftekhar, ASM and Manjunath, BS},
  		journal={arXiv preprint arXiv:2003.05541},
  		year={2020}
	}


## Results on HICO-DET and V-COCO



**Our Results on V-COCO dataset**

|Method| Scenario 1 |
|:---:|:---:|
|[InteractNet](https://arxiv.org/pdf/1704.07333.pdf)| 40.0|
|[Kolesnikov et al.](http://openaccess.thecvf.com/content_ICCVW_2019/html/SGRL/Kolesnikov_Detecting_Visual_Relationships_Using_Box_Attention_ICCVW_2019_paper.html)| 41.0|
|[GPNN](https://arxiv.org/abs/1808.07962)| 44.0 |
|[iCAN](https://arxiv.org/abs/1808.10437)| 45.3  |
|[Li et al.](https://arxiv.org/abs/1811.08264)| 47.8 |
|**VSGNet**| **51.8** |

**Our Results on HICO-DET dataset**

|Method| Full | Rare | None-Rare|
|:---:|:---:|:---:|:---:|
|[HO-RCNN](http://www-personal.umich.edu/~ywchao/publications/chao_wacv2018.pdf)| 7.81 | 5.37 | 8.54 | 
|[InteractNet](https://arxiv.org/pdf/1704.07333.pdf)|9.94 | 7.16| 10.77| 
|[GPNN](https://arxiv.org/abs/1808.07962)| 10.61  | 7.78 | 11.45 | 
|[iCAN](https://arxiv.org/abs/1808.10437)| 14.84  | 10.45 | 16.15 | 
|[Li et al.](https://arxiv.org/abs/1811.08264)| 17.03   | 13.42 | 18.11 | 
|**VSGNet**| **19.8**  | **16.05** | **20.91** | 
## Installation
1. Clone repository (recursively):
```Shell
git clone --recursive https://github.com/ASMIftekhar/VSGNet.git
```
2. Download data,annotations,object detection results:
```Shell
bash download_data.sh
```
You need to have wget and unzip packages to execute this script. Alternatively you can download the data from [here](https://drive.google.com/drive/folders/1J8mN63bNIrTdBQzq7Lpjp4qxMXgYI-yF?usp=sharing).
If you execute the script then there will be two folders in the directory "All\_data\_vcoco" and "infos". This will take close to 1.9GB space. Inside the All\_data\_vcoco folder you will find the following subdirectories.

**a.Data_vcoco**: It will contain all training and validation images of v-coco inside train2014 subdirectory and all test images of v-coco inside val2014 subdirectory.

**b.Annotations\_vcoco**: It will contain all annotations of training, validation and testing set in three json files. The annotations are taken from v-coco API and converted into our convenient format. For example, lets consider there is only one single image annotated with two verbs "smile" and "hold" along with two person and object bounding boxes. The annotation for this image will be arranged as follows:

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

3. To install all packages:
```
pip2 install -r requirements.txt
```

4. If you do not wish to move "All\_data\_vcoco" folder from the main directory then you dont need to do anything else to setup the repo. Otherwise you need to run setup.py with the location of All\_data\_vcoco. If you put it in /media/ssd2 with a new name of "data" then you need to execute the following command:
```
python2 setup.py -d /media/ssd2/data
```

## Evaluation in V-COCO
To download the pre-trained model for the results reported in the paper:
```Shell
bash download_res.sh
```
This will store the model in 'soa_paper' folder. Alternatively you can download the model from [here](https://drive.google.com/drive/folders/1J8mN63bNIrTdBQzq7Lpjp4qxMXgYI-yF?usp=sharing).

To store the best result in v-coco format run(inside "scripts/"):
```Shell
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw soa_paper -ba 8 -r t -i t
```
You can use as many gpus as you wish. Just add the necessary gpu ids in the given command.

The outputs that will be shown in the console is basically Average Precision in test set without considering bounding boxes. 

To see the results in original v-coco scheme:
```Shell
python2 calculate_map_vcoco.py -fw soa_paper -sa 34 -t test
```

## Training in V-COCO

To train the model from scratch:
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -l 0.001 -e 80 -sa 20 
```
**Flags description**:

**-fw:** Name of the folder in which the result will be stored.

**-ba:** Batch size.

**-l:** Learning rate.

**-e:** Number of epochs.

**-sa:** After how many epochs the model would be saved, remember by default for every epoch the best model will be saved. If someone wants to store the model at a particular epoch then this flag should be used.

To understand the flags more please consult main.py. The given example is a typical hyperparameter settings. The model converges normally within 40 epochs. Again,you can use as many gpus as you wish. Just add the necessary gpu ids in the given command. After running the model,  to store the results in v-coco format:
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -r t -i t
 ```
Lets consider the best result is achieved at 30th epoch then to evaluate the result in original V-COCO scheme:
```
python2 calculate_map_vcoco.py -fw new_test -sa 30 -t test
```

Code for HICO dataset will be released later. Please contact A S M Iftekhar (iftekhar@ucsb.edu) for any queries.
