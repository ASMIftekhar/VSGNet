import torch
import torch.nn as nn
import torchvision
import math
import numpy as np
import os
import helpers_preprocess as helpers_pre
import math
import json







def get_pool_loc(ims,image_id,flag_,size=(7,7), spatial_scale=1,batch_size=1):
    spatial_locs=[]
    union_box_out=[]
    pers_out=[]
    objs_out=[]

    flag='train'
    max_pool=nn.AdaptiveMaxPool2d(size)
    
    for batch in range(batch_size):
	this_image=int(image_id[batch])
	
        
        if int(flag_[batch][0])==0:
		flag='train'
	elif int(flag_[batch][0])==1:
		flag='val'
	elif int(flag_[batch][0])==2:
		flag='test'
	
        a=helpers_pre.get_compact_detections(this_image,flag)	
        roi_pers,roi_objs=a['person_bbx'],a['objects_bbx']
        union_box=helpers_pre.get_attention_maps(this_image,flag)
	union_box_out.append(torch.tensor(union_box).cuda().float())
        
        W,H,C=ims[batch].size()[1],ims[batch].size()[2],ims[batch].size()[0]
	spatial_scale=[W,H,W,H]
	image_this_batch=ims[batch]
	roi_pers=roi_pers*spatial_scale
        roi_objs=roi_objs*spatial_scale
        
        ##### Pooling Persons ##########
        for index,roi_val in enumerate (roi_pers):
	    x1,y1,x2,y2=int(roi_val[0]),int(roi_val[1]),int(roi_val[2]),int(roi_val[3])
	    sp=[x1,y1,x2,y2,x2-x1,y2-y1] 
	    im = image_this_batch.narrow(0, 0, image_this_batch.size()[0])[..., y1:(y2+1), x1:(x2+1)]
            pooled=max_pool(im)
	    pers_out.append((pooled))
	    spatial_locs.append(sp)
        
        
        ### Pooling Objects #####
        for index,roi_val in enumerate (roi_objs):
	    x1,y1,x2,y2=int(roi_val[0]),int(roi_val[1]),int(roi_val[2]),int(roi_val[3])
	    sp=[x1,y1,x2,y2,x2-x1,y2-y1] 
	    im = image_this_batch.narrow(0, 0, image_this_batch.size()[0])[..., y1:(y2+1), x1:(x2+1)]
            pooled=max_pool(im)
	    objs_out.append((pooled))
	    spatial_locs.append(sp)
    #import pdb;pdb.set_trace()
    return torch.stack(pers_out),torch.stack(objs_out),spatial_locs,torch.cat(union_box_out)

def extract_spatial(hum_box,obj_box):
	x1h,y1h,x2h,y2h,wh,hh=float(hum_box[0]),float(hum_box[1]),float(hum_box[2]),float(hum_box[3]),float(hum_box[4]),float(hum_box[5])
	x1o,y1o,x2o,y2o,wo,ho=float(obj_box[0]),float(obj_box[1]),float(obj_box[2]),float(obj_box[3]),float(obj_box[4]),float(obj_box[5])
	if wh==0.0:
		wh+=1
	if hh==0.0:
		hh+=1
	diff_x=0.001 if x1h-x1o==0 else x1h-x1o
	diff_y=0.001 if y1h-y1o==0 else y1h-y1o
	
	if wo!=0 and ho!=0:	
		extract=torch.FloatTensor([diff_x/wo,diff_y/ho,math.log(wh/wo),math.log(hh/ho)])
	elif wo==0 and ho!=0:	
		extract=torch.FloatTensor([diff_x,diff_y/ho,math.log(wh),math.log(hh/ho)])
	elif wo!=0 and ho==0:	
		extract=torch.FloatTensor([diff_x/wo,diff_y,math.log(wh/wo),math.log(hh)])
	else:
		extract=torch.FloatTensor([diff_x,diff_y,math.log(wh),math.log(hh)])
	return extract.cuda()

def pairing(pers,objs,context,spatial_locs,pairs_info):    
    
    pairs_out=[]
    pers_out=[]
    objs_out=[]
    
    
    start=0
    start_p=0
    start_o=0
    for batch in range(len(pairs_info)):
	
	this_batch_per=int(pairs_info[batch][0])
	this_batch_obj=int(pairs_info[batch][1])	
	this_batch_len=int(pairs_info[batch][0]+pairs_info[batch][1])
	batch_pers,batch_objs=pers[start_p:start_p+this_batch_per],objs[start_o:start_o+this_batch_obj]
	batch_context=context[batch]
	sp_locs_batch=spatial_locs[start:start+this_batch_len]
	sp_locs_pers_batch,sp_locs_objs_batch=sp_locs_batch[0:this_batch_per],sp_locs_batch[this_batch_per:this_batch_per+this_batch_obj]
	pers_objs=[]
        for ind_p,i in enumerate(batch_pers):

            for ind_o,j in enumerate(batch_objs):
                sp_features=extract_spatial(sp_locs_pers_batch[ind_p],sp_locs_objs_batch[ind_o])
                pers_objs.append(torch.cat([i,j,sp_features],0))
                #pers_objs.append(torch.cat([i,j],0))
        
        pers_objs_batch=torch.stack(pers_objs)
        
        pairs_out.append(torch.cat([pers_objs_batch,batch_context.repeat(pers_objs_batch.size()[0],1)],1))
	pers_out.append(batch_pers)	
	objs_out.append(batch_objs)	
	
        start+=this_batch_len
	start_p+=this_batch_per
	start_o+=this_batch_obj
    
    
    return torch.cat(pairs_out),torch.cat(pers_out),torch.cat(objs_out)
