import numpy as np
import pickle
import cv2
import pandas as pd
import json
import helpers_preprocess as helpers_pre 

with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)  


OBJ_PATH_train_s = all_data_dir+'Object_Detections_vcoco/train/'  
OBJ_PATH_test_s= all_data_dir+'Object_Detections_vcoco/val/' 
image_dir_train=all_data_dir+'Data_vcoco/train2014' 
image_dir_val=all_data_dir+'Data_vcoco/train2014' 
image_dir_test=all_data_dir+'Data_vcoco/val2014' 


VERB2ID = {u'carry': 0,
 u'catch': 1,
 u'cut_instr':2,
 u'cut_obj': 3,
 u'drink': 4,
 u'eat_instr':5,
 u'eat_obj': 6,
 u'hit_instr':7,
 u'hit_obj': 8,
 u'hold': 9,
 u'jump': 10, 
 u'kick': 11, 
 u'lay': 12, 
 u'look': 13, 
 u'point': 14, 
 u'read': 15, 
 u'ride': 16, 
 u'run': 17, 
 u'sit': 18, 
 u'skateboard': 19, 
 u'ski': 20, 
 u'smile': 21, 
 u'snowboard': 22, 
 u'stand': 23, 
 u'surf': 24, 
 u'talk_on_phone': 25, 
 u'throw': 26, 
 u'walk': 27, 
 u'work_on_computer': 28
}

ID2VERB=dict((VERB2ID[i],i) for i in VERB2ID)

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

pd.options.display.max_columns = 250  # None -> No Restrictions
pd.options.display.max_rows = 200    # None -> Be careful with



def visual(image_id,flag,pairs_info,score_HOI,score_interact,score_obj_box,score_per_box,score_REL,score_HOI_pair,ground_truth):
	start=0
	for batch in range(len(image_id)):
	    this_image=int(image_id[batch])
	    a=helpers_pre.get_compact_detections(this_image,flag)
	    person_bbxn=a['person_bbx']
	    obj_bbxn=a['objects_bbx']
	    this_batch_pers=int(pairs_info[batch][0])
	    this_batch_objs=int(pairs_info[batch][1])
	    increment=this_batch_pers*this_batch_objs
            ground_truth_this_batch=ground_truth[start:start+increment]
	    score_HOI_this_batch=score_HOI[start:start+increment]
	    start+=increment
	    if flag=='train':
    
                cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (this_image)
        	
                image_dir_s=image_dir_train+'/COCO_train2014_%.12i.jpg'%(this_image)

            elif flag=='test':
    
                cur_obj_path_s = OBJ_PATH_test_s + "COCO_val2014_%.12i.json" % (this_image)
                image_dir_s=image_dir_test +'/COCO_val2014_%.12i.jpg'%(this_image)
            elif flag=='val':
                cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (this_image)
        	image_dir_s=image_dir_val+ '/COCO_train2014_%.12i.jpg'%(this_image)
	    with open(cur_obj_path_s) as fp: 
	        detections = json.load(fp)
	    img_H = detections['H']
            img_W = detections['W']
	    person_bbx= np.array([img_W,img_H,img_W,img_H],dtype=float)*person_bbxn
            obj_bbx= np.array([img_W,img_H,img_W,img_H],dtype=float)*obj_bbxn
            img = cv2.imread(image_dir_s,3)
	    start_index=0
	    for person_box in person_bbx:
	        for object_box in obj_bbx:
		    ground_truth_this_sample=ground_truth_this_batch[start_index]
		    score_HOI_this_sample=score_HOI_this_batch[start_index]
	            print(score_HOI_this_sample)			    
		    pred=[('GROUND_TRUTH',[(ID2VERB[ind], float("%.2f" % ground_truth_this_sample[ind])) for ind in np.argsort(ground_truth_this_sample)[-5:][::-1]])]
		    pred.append(('TOTAL_PREDICTION',[(ID2VERB[ind], float("%.2f" % score_HOI_this_sample[ind])) for ind in np.argsort(score_HOI_this_sample)[-5:][::-1]]))
		    prediction=pd.DataFrame(pred,columns =['Name', 'Prediction'])
		    
                    img = cv2.imread(image_dir_s,3)
		    x, y, w, h = int(person_box[0]),int(person_box[1]),int(person_box[2]-person_box[0]),int(person_box[3]-person_box[1])
		    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
		    x, y, w, h = int(object_box[0]),int(object_box[1]),int(object_box[2]-object_box[0]),int(object_box[3]-object_box[1])
		    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
			
                    print('\nPredictions (Five Highest Confidence Class):\n{}\n'.format(prediction))

		    cv2.imshow('image',img)
	            start_index+=1		
		    k=cv2.waitKey(0)
		    if k == 27:         # wait for ESC key to exit

		        cv2.destroyAllWindows()
			

		    
		if k == 27:         # wait for ESC key to exit

		    cv2.destroyAllWindows()
	    if k == 27:         # wait for ESC key to exit

	        cv2.destroyAllWindows()

	cv2.destroyAllWindows()
	
