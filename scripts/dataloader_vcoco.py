from __future__ import print_function, division
import json
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import helpers_preprocess as labels
from PIL import Image


bad_detections_train,bad_detections_val,bad_detections_test=labels.dry_run()
#bad_detections_train,bad_detections_val,bad_detections_test=[],[],[]
NO_VERB=29
def vcoco_collate(batch):
    image =[] 
    image_id=[]
    pairs_info=[]
    labels_all=[]
    labels_single=[]
    for index,item in enumerate(batch):
        image.append(item['image'])
        image_id.append(torch.tensor(int(item['image_id'])))
        pairs_info.append(torch.tensor(np.shape(item['labels_all'])))
        tot_HOI=int(np.shape(item['labels_single'])[0])
        labels_all.append(torch.tensor(item['labels_all'].reshape(tot_HOI,NO_VERB)))
        labels_single.append(torch.tensor(item['labels_single']))
    return [torch.stack(image),torch.cat(labels_all),torch.cat(labels_single),torch.stack(image_id),torch.stack(pairs_info)]

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
    


        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img2 = transform.resize(image, (new_h, new_w))

        return img2



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        

        	
        image = image.transpose((2, 0, 1))
       
        return torch.from_numpy(image).float()


class vcoco_Dataset(Dataset):
    

    def __init__(self, json_file_image,root_dir,transform=None):
	with open(json_file_image) as json_file_:               
            self.vcoco_frame_file = json.load(json_file_)
	self.flag=json_file_image.split('/')[-1].split('_')[0]
	if self.flag=='train':
		self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_train)]  
	elif self.flag=='val':
		self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_val)]	
	elif self.flag=='test':
		self.vcoco_frame= [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_test)]	
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.vcoco_frame)

    def __getitem__(self, idx):
	if self.flag=='test':
		img_pre_suffix='COCO_val2014_'+str(self.vcoco_frame[idx]).zfill(12)+'.jpg'
	else:
	       img_pre_suffix='COCO_train2014_'+str(self.vcoco_frame[idx]).zfill(12)+'.jpg'
	all_labels=labels.get_compact_label(int(self.vcoco_frame[idx]),self.flag)
	labels_all=all_labels['labels_all']
	labels_single=all_labels['labels_single']
        
	img_name = os.path.join(self.root_dir,img_pre_suffix)
     	ids=[int(self.vcoco_frame[idx]),self.flag]   
	image=Image.open(img_name).convert('RGB')
	image=np.array(image)
	
        if self.transform:
            image = self.transform(image)
        sample = {'image':image ,'labels_all':labels_all,'labels_single':labels_single,'image_id':self.vcoco_frame[idx]}
        return sample
