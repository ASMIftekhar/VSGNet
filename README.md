# VSGNet 
### [**VSGNet:Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ulutan_VSGNet_Spatial_Attention_Network_for_Detecting_Human_Object_Interactions_Using_CVPR_2020_paper.pdf) 

[Oytun Ulutan*](https://sites.google.com/view/oytun-ulutan), [A S M Iftekhar*](https://sites.google.com/view/asmiftekhar/home), [B S Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath).

Official repository of our [**CVPR 2020**](http://cvpr2020.thecvf.com/) paper.

![Overview of VSGNET](https://github.com/ASMIftekhar/VSGNet/blob/master/7850-teaser.gif?raw=true)
## Citing
If you find this work useful, please consider our paper to cite:

	 @InProceedings{Ulutan_2020_CVPR,
	author = {Ulutan, Oytun and Iftekhar, A S M and Manjunath, B. S.},
	title = {VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions},
	booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
	}



## Results on HICO-DET and V-COCO



## Our Results on V-COCO dataset

|Method| mAP (Scenario 1)|
|:---:|:---:|
|[InteractNet](https://arxiv.org/pdf/1704.07333.pdf)| 40.0|
|[Kolesnikov et al.](http://openaccess.thecvf.com/content_ICCVW_2019/html/SGRL/Kolesnikov_Detecting_Visual_Relationships_Using_Box_Attention_ICCVW_2019_paper.html)| 41.0|
|[GPNN](https://arxiv.org/abs/1808.07962)| 44.0 |
|[iCAN](https://arxiv.org/abs/1808.10437)| 45.3  |
|[Li et al.](https://arxiv.org/abs/1811.08264)| 47.8 |
|[**VSGNet**](https://arxiv.org/abs/2003.05541)| **51.8** |

## Our Results on HICO-DET dataset

**Object Detector Pre-trained on COCO**
|Method| mAP (Full) | mAP (Rare) | mAP (None-Rare)|
|:---:|:---:|:---:|:---:|
|[HO-RCNN](http://www-personal.umich.edu/~ywchao/publications/chao_wacv2018.pdf)| 7.81 | 5.37 | 8.54 | 
|[InteractNet](https://arxiv.org/pdf/1704.07333.pdf)|9.94 | 7.16| 10.77| 
|[GPNN](https://arxiv.org/abs/1808.07962)| 10.61  | 7.78 | 11.45 | 
|[iCAN](https://arxiv.org/abs/1808.10437)| 14.84  | 10.45 | 16.15 | 
|[Li et al.](https://arxiv.org/abs/1811.08264)| 17.03   | 13.42 | 18.11 | 
|[**VSGNet**](https://arxiv.org/abs/2003.05541)| **19.8**  | **16.05** | **20.91** | 

**Object Detector Fine-Tuned on HICO**

We use the object detection results from [DRG](https://github.com/vt-vl-lab/DRG).
|Method| mAP (Full) | mAP (Rare) | mAP (None-Rare)|
|:---:|:---:|:---:|:---:|
|[UniDet](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600494.pdf)|17.58 |11.72 |19.33 |
|[IP-Net](https://arxiv.org/pdf/2003.14023.pdf) | 19.56 |12.79| 21.58 |
|[PPDM](https://arxiv.org/pdf/1912.12898v1.pdf) |21.10 |14.46| 23.09| 
|[Functional](https://arxiv.org/pdf/1904.03181.pdf) |21.96 |16.43|23.62| 
|[VCL](https://github.com/zhihou7/VCL)|23.63 |17.21 |25.55 |
|[ConsNet](https://github.com/YLiuEric/ConsNet)|24.39 |17.10 |26.56|
|[DRG](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570681.pdf)|24.53 |19.47 |26.04 |
|[IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network))|26.29|**22.61**|27.39|
|[**VSGNet**](https://arxiv.org/abs/2003.05541)| **26.54**| 21.26 | **28.12** |


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
If you execute the script then there will be two folders in the directory "All\_data" and "infos". This will take close to 10GB space. This contains both of the datasets and all the essential files. Also, if you just want to work with v-coco, download "All_data_vcoco" from the link.  

Inside the All\_data folder you will find the following subdirectories.

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

**e.Data_hico**: It will contain all the training images of HICO-DET inside train2015 subdirectory and all test images of HICO_DET inside test2015 subdirectory.

**f.Annotations\_hico**: same as folder (b) but for HICO_DET dataset.

**g.Object\_Detections\_hico**: same as folder (c) but for HICO_DET dataset.

**h.bad\_Detections\_hico**: It will contain the list of images in HICO_DET dataset where our object detector fails to detect any person or object.

**j.hico\_infos**: It will contain additional files required to run training and testing in HICO_DET.

3. To install all packages (preferable to run in a python2 virtual environment):
```
pip2 install -r requirements.txt
```
For HICO_DET evaluation we will use python3 environment, to install those packages (preferable to run in a python3 virtual environment):
```
pip3 install -r requirements3.txt
```
Run only compute_map.sh in a python 3 enviornment. For all other use python 2 environment.

4. If you do not wish to move "All\_data" folder from the main directory then you dont need to do anything else to setup the repo. Otherwise you need to run setup.py with the location of All\_data. If you put it in /media/ssd2 with a new name of "data" then you need to execute the following command:
```
python2 setup.py -d /media/ssd2/data/
```

## Downloading the Pre-Trained Models:
To download the pre-trained models for the results reported in the paper:
```Shell
bash download_res.sh
```
This will store the model for v-coco in 'soa_paper' folder and the model for HICO_DET in 'soa_paper_hico'. Alternatively you can download the models from [here](https://drive.google.com/drive/folders/1J8mN63bNIrTdBQzq7Lpjp4qxMXgYI-yF?usp=sharing).

## Evaluation in V-COCO


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
## Evaluation in HICO_DET


To store the best result in HICO_DET format run (inside "scripts_hico/"):
```Shell
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw soa_paper_hico -ba 8 -r t -i t
```
You can use as many gpus as you wish. Just add the necessary gpu ids in the given command.

The outputs that will be shown in the console is basically Average Precision in test set without considering bounding boxes. 

To see the results in original HICO_DET scheme run (inside "scripts_hico/HICO_eval/")
```Shell
bash compute_map.sh soa_paper_hico 20
```
The evaluation code has been adapted from the [No-Frills repository.](https://github.com/BigRedT/no_frills_hoi_det)Here, 20 indicates the number of cpu cores to be used for evaluation, this can be changed to any number based on the system. 
## Training in V-COCO

To train the model from scratch (inside "scripts/"):
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -l 0.001 -e 80 -sa 20 
```
**Flags description**:

**-fw:** Name of the folder in which the result will be stored.

**-ba:** Batch size.

**-l:** Learning rate.

**-e:** Number of epochs.

**-sa:** After how many epochs the model would be saved, remember by default for every epoch the best model will be saved. If someone wants to store the model at a particular epoch then this flag should be used.

To understand the flags more please consult main.py. The given example is a typical hyperparameter settings. The model converges normally within 40 epochs. Again,you can use as many gpus as you wish. Just add the necessary gpu ids in the given command. After running the model,  to store the results in v-coco format (inside "scripts/"):
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -r t -i t
 ```
Lets consider the best result is achieved at 30th epoch then to evaluate the result in original V-COCO scheme(inside "scripts/"):
```
python2 calculate_map_vcoco.py -fw new_test -sa 30 -t test
```
## Training in HICO_DET

To train the model from scratch (inside "scripts_hico/"):
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -l 0.001 -e 80 -sa 20 
```
The flags are same as v-coco. The model converges normally within 30 epochs. Again,you can use as many gpus as you wish. Just add the necessary gpu ids in the given command. We have used 4 2080Tis to train HICO_DET with a batch size of 8 per gpu. It takes around 40 minutes per epoch.  
After running the model, to store the results in HICO_DET format (inside "scripts_hico/"):
```
CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -r t -i t
```
To evaluate the result in original HICO_DET scheme (inside "scripts_hico/HICO_eval/"):
```Shell
bash compute_map.sh new_test 20
```

Please contact A S M Iftekhar (iftekhar@ucsb.edu) for any queries.
