import torch
import torch.nn as nn
import time
import errno
import os
import gc
import pickle
import shutil
import json
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import calculate_ap_classwise as ap
import matplotlib.pyplot as plt
import random

import helpers_preprocess as helpers_pre
import pred_vis as viss
import prior_vcoco as prior
import proper_inferance_file as proper
from tqdm import tqdm


sigmoid=nn.Sigmoid()

### Fixing Seeds#######
device = torch.device("cuda")
seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
softmax=nn.Softmax()
##########################################


########## Paramets for person to object class mapping#####
SCORE_TH=0.6
SCORE_OBJ=0.3
epoch_to_change=400
thresh_hold=-1
##############################################################

#### Loss functions Definition########
loss_com = nn.BCEWithLogitsLoss(reduction='sum')
loss_com_class = nn.BCEWithLogitsLoss(reduction='none')
loss_com_combine = nn.BCELoss(reduction='none')
loss_com_single = nn.BCEWithLogitsLoss(reduction='sum')
##################################
#import pdb;pdb.set_trace()
no_of_classes=29



##Helper Functions##
### Fixing the seed for all threads#####
def _init_fn(worker_id):
    np.random.seed(int(seed))
#######Extending Number of People############
def extend(inputt,extend_number):
        #import pdb;pdb.set_trace()
        res=np.zeros([1,np.shape(inputt)[-1]])
        for a in inputt:
                x=np.repeat(a.reshape(1,np.shape(inputt)[-1]),extend_number,axis=0)
                res=np.concatenate([res,x],axis=0)
        #import pdb;pdb.set_trace()
        return res[1:]
######################################
####### Extening Number of Objects##########
def extend_object(inputt,extend_number):
        #import pdb;pdb.set_trace()
        res=np.zeros([1,np.shape(inputt)[-1]])
        #import pdb;pdb.set_trace()
        x=np.array(inputt.tolist()*extend_number)
                #import pdb;pdb.set_trace()
        res=np.concatenate([res,x],axis=0)
        #import pdb;pdb.set_trace()
        return res[1:]
#############################################

############## Filtering results for preparing the output as per as v-coco###############################
def filtering(predicted_HOI,true,persons_np,objects_np,filters,pairs_info,image_id):
	 res1=np.zeros([1,no_of_classes])
	 res2=np.zeros([1,no_of_classes])
	 res3=np.zeros([1,no_of_classes])
	 res4=np.zeros([1,4])
	 res5=np.zeros([1,4])
	 dict1={}
	 a=0
	 increment=[int(i[0]*i[1]) for i in pairs_info]
	 #import pdb;pdb.set_trace()
	 start=0
	 for index,i in enumerate(filters):
		
		res1=np.concatenate([res1,predicted_HOI[index].reshape(1,no_of_classes)],axis=0)
		res2=np.concatenate([res2,true[index].reshape(1,no_of_classes)],axis=0)
		res3=np.concatenate([res3,predicted_HOI[index].reshape(1,no_of_classes)],axis=0)
		res4=np.concatenate([res4,persons_np[index].reshape(1,4)],axis=0)
		res5=np.concatenate([res5,objects_np[index].reshape(1,4)],axis=0)
		if index==start+increment[a]-1:
	 		#import pdb;pdb.set_trace()
			dict1[int(image_id[a]),'score']=res3[1:]
			dict1[int(image_id[a]),'pers_bbx']=res4[1:]
			dict1[int(image_id[a]),'obj_bbx']=res5[1:]
			res3=np.zeros([1,no_of_classes])
	 		res4=np.zeros([1,4])
	 		res5=np.zeros([1,4])
			start+=increment[a]
			a+=1
	 		#import pdb;pdb.set_trace()
	 return dict1
#### Saving CheckPoint##########
def save_checkpoint(state,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
###################################
### LIS function from https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network##########
def LIS(x,T,k,w):		
	return T/(1+np.exp(k-w*x)) 
#####################################################################################################




def train_test(model, optimizer,scheduler,dataloader,number_of_epochs,break_point,saving_epoch,folder_name,batch_size,infr,start_epoch,mean_best,visualize):
    

#### Creating the folder where the results would be stored##########

    try:
	os.mkdir(folder_name)
    except OSError as exc:
    	if exc.errno != errno.EEXIST:
        	raise
    	pass
    file_name=folder_name+'/'+'result.pickle'
####################################################################


    loss_epoch_train=[]
    loss_epoch_val=[]
    loss_epoch_test=[]
    initial_time = time.time()
    result=[]

   ##### Freeing out the cache memories from gpus and declaring the phases######
    torch.cuda.empty_cache()
    phases=['train','val','test']	
    
    if infr=='t' and visualize=='f':### If running from a pretrained model only for saving best result ###### 
	start_epoch=start_epoch-1
	phases=['test']
        end_of_epochs=start_epoch+1
	print('Only doing testing for storing result from a model')


    elif visualize!='f':
        if visualize not in phases:
            print("ERROR! Asked to show result from a unknown set.The choice should be among train,val,test")
            return
        else:    
            phases=[visualize]
            end_of_epochs=start_epoch+1
	    print('Only showing predictions from a model')
    else:
        end_of_epochs=start_epoch+number_of_epochs
   ##### Starting the Epochs#############
    for epoch in range(start_epoch,end_of_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch+1,end_of_epochs))
        print('-' * 10)
        #print('Lr: {}'.format(scheduler.get_lr()))
	initial_time_epoch=time.time()
	

	


	for phase in phases:
    		if phase=='train':
			model.train()
		elif phase=='val':
			model.train()
		else:
			model.eval()


                print('In {}'.format(phase))
		detections_train=[]
		detections_val=[]
		detections_test=[] 	
		


                true_scores_class=np.ones([1, 80], dtype=int)
                true_scores=np.ones([1, 29], dtype=int)
                true_scores_single=np.ones([1, 1], dtype=int)
                predicted_scores=np.ones([1, 29], dtype=float)
                predicted_scores_single=np.ones([1, 1], dtype=float)
                predicted_scores_class=np.ones([1, 80], dtype=float)
		acc_epoch=0
		iteration=1
		

    		torch.cuda.empty_cache()
		####Starting the iterations##################
		for iterr,i in enumerate(tqdm(dataloader[phase])):
			if iterr%20==0:
    				torch.cuda.empty_cache()
								        
			inputs=i[0].to(device)
			labels=i[1].to(device)
			labels_single=i[2].to(device)
                        image_id=i[3]
			pairs_info=i[4]
			minbatch_size=len(pairs_info)

			optimizer.zero_grad()
    			if phase=='train':
				nav=torch.tensor([[0,epoch]]*minbatch_size).to(device)
			elif phase=='val':
				nav=torch.tensor([[1,epoch]]*minbatch_size).to(device)
			else:
				nav=torch.tensor([[2,epoch]]*minbatch_size).to(device)


			#import pdb;pdb.set_trace()		
			true=(labels.data).cpu().numpy()
			true_single=(labels_single.data).cpu().numpy()
			





			with torch.set_grad_enabled(phase=='train'or phase=='val'):
				model_out = model(inputs,pairs_info,pairs_info,image_id,nav,phase)
				outputs=model_out[0]
				outputs_single=model_out[1]
				outputs_combine=model_out[2]
				outputs_gem=model_out[3]
				#outputs_pose=model_out[7]
				

				predicted_HOI=sigmoid(outputs).data.cpu().numpy()
				predicted_HOI_combine=sigmoid(outputs_combine).data.cpu().numpy()
				predicted_single=sigmoid(outputs_single).data.cpu().numpy()	    
				predicted_gem=sigmoid(outputs_gem).data.cpu().numpy()	    
				predicted_HOI_pair=predicted_HOI 
				#predicted_HOI_pose=sigmoid(outputs_pose).data.cpu().numpy()
				

				
 
				start_index=0
				start_obj=0
				start_pers=0
				start_tot=0 
				pers_index=1
				persons_score_extended=np.zeros([1,1])
				objects_score_extended=np.zeros([1,1])
				class_ids_extended=np.zeros([1,1])
				persons_np_extended=np.zeros([1,4])
				objects_np_extended=np.zeros([1,4])  
				start_no_obj=0
				class_ids_total=[]



				############# Extending Person and Object Boxes and confidence scores to Multiply with all Pairs########## 
				for batch in range(len(pairs_info)):
				
					persons_score=[]
					objects_score=[]
					class_ids=[]
					objects_score.append(float(1))
			
					this_image=int(image_id[batch])
					scores_total=helpers_pre.get_compact_detections(this_image,phase)
					persons_score,objects_score,persons_np,objects_np,class_ids=scores_total['person_bbx_score'],scores_total['objects_bbx_score'],scores_total['person_bbx'],scores_total['objects_bbx'],scores_total['class_id_objects']	
					temp_scores=extend(np.array(persons_score).reshape(len(persons_score),1),int(pairs_info[batch][1]))
					persons_score_extended=np.concatenate([persons_score_extended,temp_scores])
					temp_scores=extend(persons_np,int(pairs_info[batch][1]))
					persons_np_extended=np.concatenate([persons_np_extended,temp_scores])
					temp_scores=extend_object(np.array(objects_score).reshape(len(objects_score),1),int(pairs_info[batch][0]))
					objects_score_extended=np.concatenate([objects_score_extended,temp_scores])
					temp_scores=extend_object(objects_np,int(pairs_info[batch][0]))
					objects_np_extended=np.concatenate([objects_np_extended,temp_scores])
					temp_scores=extend_object(np.array(class_ids).reshape(len(class_ids),1),int(pairs_info[batch][0]))
					class_ids_extended=np.concatenate([class_ids_extended,temp_scores])
					class_ids_total.append(class_ids)
						
					start_pers=start_pers+int(pairs_info[batch][0])
					start_obj=start_obj+int(pairs_info[batch][1])
					start_tot=start_tot+int(pairs_info[batch][1])*int(pairs_info[batch][0])
					###################################################################################################################
				
				#### Applying LIS#######
				persons_score_extended=LIS(persons_score_extended,8.3,12,10)
				objects_score_extended=LIS(objects_score_extended,8.3,12,10)
				##################################
					
				##### Multiplying the score from different streams along with the prior function from ican##########
				predicted_HOI=predicted_HOI*predicted_HOI_combine*predicted_single*predicted_gem*objects_score_extended[1:]*persons_score_extended[1:]
				loss_mask=prior.apply_prior(class_ids_extended[1:],predicted_HOI)
				predicted_HOI=loss_mask*predicted_HOI
			
				#### Calculating Loss############
				N_b=minbatch_size*29#*int(total_elements[0])#*29 #pairs_info[1]*pairs_info[2]*pairs_info[3]
				hum_obj_mask=torch.Tensor(objects_score_extended[1:]*persons_score_extended[1:]*loss_mask).cuda()
				lossf=torch.sum(loss_com_combine(sigmoid(outputs)*sigmoid(outputs_combine)*sigmoid(outputs_single)*hum_obj_mask*sigmoid(outputs_gem),labels.float()))/N_b
				lossc=lossf.item()			

				acc_epoch+=lossc
				iteration+=1
				if phase=='train' or phase=='val':#### Flowing the loss backwards#########
					lossf.backward()
					optimizer.step()


				###########################################################
				del lossf
				del model_out
				del inputs
				del outputs
				del labels
			####### If we want to do Visualization#########	 
			if visualize!='f' :
				viss.visual(image_id,phase,pairs_info,predicted_HOI,predicted_single,objects_score_extended[1:],persons_score_extended[1:],predicted_HOI_combine,predicted_HOI_pair,true)
				
				
			#####################################################################
			
			##### Preparing for Storing Results##########	
			predicted_scores=np.concatenate((predicted_scores,predicted_HOI),axis=0)
			true_scores=np.concatenate((true_scores,true),axis=0)
			predicted_scores_single=np.concatenate((predicted_scores_single,predicted_single),axis=0)
			true_scores_single=np.concatenate((true_scores_single,true_single),axis=0)
			#############################################


			#### Storing the result in V-COCO Format##########
			if phase=='test':
			    if (epoch+1)%saving_epoch==0 or infr=='t':
                                all_scores=filtering(predicted_HOI,true,persons_np_extended[1:],objects_np_extended[1:],predicted_single,pairs_info,image_id)
				#prep.infer_format(image_id,all_scores,phase,detections_test,pairs_info)
				proper.infer_format(image_id,all_scores,phase,detections_test,pairs_info)
			######################################################		



					




			## Breaking in particular number of epoch####
			if iteration==break_point+1:
		
			      		break;
			#############################################

		if phase=='train':
			loss_epoch_train.append((acc_epoch))
   			AP,AP_single=ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                        AP_train = pd.DataFrame(AP,columns =['Name_TRAIN', 'Score_TRAIN'])
                        AP_train_single = pd.DataFrame(AP_single,columns =['Name_TRAIN', 'Score_TRAIN'])
       			
		elif phase=='val':
			loss_epoch_val.append((acc_epoch))
   			AP,AP_single=ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                        AP_val = pd.DataFrame(AP,columns =['Name_VAL', 'Score_VAL'])
                        AP_val_single = pd.DataFrame(AP_single,columns =['Name_VAL', 'Score_VAL'])

		elif phase=='test':
			loss_epoch_test.append((acc_epoch))
   			AP,AP_single=ap.class_AP(predicted_scores[1:,:],true_scores[1:,:],predicted_scores_single[1:,],true_scores_single[1:,])
                        AP_test = pd.DataFrame(AP,columns =['Name_TEST', 'Score_TEST'])
                        AP_test_single = pd.DataFrame(AP_single,columns =['Name_TEST', 'Score_TEST'])
			if (epoch+1)%saving_epoch==0 or infr=='t':
				file_name_p=folder_name+'/'+'test{}.pickle'.format(epoch+1)   
                                with open(file_name_p, 'wb') as handle:     
                                 	pickle.dump(detections_test, handle)
	
	
	
        ###### Saving the Model###########
	mean=AP_test.to_records(index=False)[29][1]	
	####Best Model######
	if mean>mean_best and infr!='t':
		mean_best=mean
		save_checkpoint({
            		'epoch': epoch + 1,
            		'state_dict': model.state_dict(),
            		'mean_best': mean_best,
            		'optimizer' : optimizer.state_dict(),
	    		'scheduler':scheduler.state_dict()
        	},filename=folder_name+'/'+'bestcheckpoint.pth.tar')
	###############################
	
	if (epoch+1)%saving_epoch==0  and infr!='t':
		
		save_checkpoint({
            		'epoch': epoch + 1,
            		'state_dict': model.state_dict(),
            		'mean_best': mean_best,
            		'optimizer' : optimizer.state_dict(),
	    		'scheduler':scheduler.state_dict()
        	},filename=folder_name+'/'+str(epoch + 1)+'checkpoint.pth.tar')
	#####################################

	if infr=='t':	
	
		AP_final=pd.concat([AP_test],axis=1)
		AP_final_single=pd.concat([AP_test_single],axis=1)
	        result.append(AP_final)		
	else:
		AP_final=pd.concat([AP_train,AP_val,AP_test],axis=1)
		AP_final_single=pd.concat([AP_train_single,AP_val_single,AP_test_single],axis=1)
       	        ##### This file will store each epoch result in a pickle format####
                with open(file_name, 'wb') as handle:                                                
	            pickle.dump(result, handle)
	time_elapsed = time.time() - initial_time_epoch
	print('APs in EPOCH:{}'.format(epoch+1))
	print(AP_final)	
	print(AP_final_single)
	try:	
    		print('Loss_train:{},Loss_validation:{},Loss_test:{}'.format(loss_epoch_train[epoch-start_epoch],loss_epoch_val[epoch-start_epoch],loss_epoch_test[epoch-start_epoch]))
	except:
    		print('Loss_test:{}'.format(loss_epoch_test[epoch-start_epoch]))

		
	print('This epoch completes in {:.0f}m {:.06f}s'.format(
		      	        time_elapsed // 60, time_elapsed % 60))
	if infr=='t':
		break
	
                
    time_elapsed = time.time() - initial_time
    print('The whole process runs for {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed //3600, (time_elapsed % 3600) // 60,((time_elapsed % 3600)%60)%60))
    return 
