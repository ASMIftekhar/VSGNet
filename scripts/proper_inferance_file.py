import json
import torch
import numpy as np


with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)  
OBJ_PATH_train_s = all_data_dir+'Object_Detections_vcoco/train/'
OBJ_PATH_test_s= all_data_dir+'Object_Detections_vcoco/val/'
number_of_roles=[2,2,2,2,2,2,2,2,2,2,2,2,2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2]


proper_keys=['carry_agent', 'carry_obj', 'catch_agent', 'catch_obj', 'cut_agent', 'cut_instr','cut_agent', 'cut_obj', 'drink_agent', 'drink_instr', 'eat_agent', 'eat_instr','eat_agent','eat_obj', 'hit_agent', 'hit_instr','hit_agent','hit_obj', 'hold_agent', 'hold_obj', 'jump_agent', 'jump_instr', 'kick_agent', 'kick_obj', 'lay_agent', 'lay_instr', 'look_agent', 'look_obj', 'point_agent', 'point_instr', 'read_agent', 'read_obj', 'ride_agent', 'ride_instr', 'run_agent', 'sit_agent', 'sit_instr', 'skateboard_agent', 'skateboard_instr', 'ski_agent', 'ski_instr', 'smile_agent', 'snowboard_agent', 'snowboard_instr', 'stand_agent', 'surf_agent', 'surf_instr', 'talk_on_phone_agent', 'talk_on_phone_instr', 'throw_agent', 'throw_obj', 'walk_agent', 'work_on_computer_agent', 'work_on_computer_instr']
def infer_format(image_id,all_scores_batch,flag,all_detections,pairs_info):
	this_batch_start=0
	for batch in range(len(image_id)):
	    this_image=int(image_id[batch])
	    persons=all_scores_batch[this_image,'pers_bbx']
	    
	    objects=all_scores_batch[this_image,'obj_bbx']
	    hum_scores=0
	    this_batch_pers=int(pairs_info[batch][0])
	    this_batch_objs=int(pairs_info[batch][1])
	    increment=this_batch_pers*this_batch_objs
	    all_scores=all_scores_batch[this_image,'score']
	    if flag=='train':
		    
		    cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (this_image)

	    elif flag=='test':
		    
		    cur_obj_path_s = OBJ_PATH_test_s + "COCO_val2014_%.12i.json" % (this_image)
	    elif flag=='val':
		    
		    cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (this_image)
	    with open(cur_obj_path_s) as fp:
	        detections = json.load(fp)
	    persons_score=[]
	    objects_score=[]
	    objects_score.append(float(1))	
	    number_of_objects=len(objects)
	    persons_score=np.array(persons_score,dtype=float)
	    objects_score=np.array(objects_score,dtype=float)
	    img_H = detections['H']
	    img_W = detections['W']
	    index_person=0
	    infer_dict={}
	    for item_no,role_ids in enumerate((all_scores)):
	        person_bbxn=persons[item_no]
	        obj_bbxn=objects[item_no]
	        person_bbx= np.array([img_W,img_H,img_W,img_H],dtype=float)*person_bbxn
	        obj_bbx= np.array([img_W,img_H,img_W,img_H],dtype=float)*obj_bbxn
	        infer_dict={}
	    
	        infer_dict['person_box']=person_bbx.tolist()
	        infer_dict['image_id']=this_image
	        dict_index=0
	        for index,k in (enumerate(role_ids)):
		    person_action_score=k#*person_confidence
		    instances=number_of_roles[index]
                    for j in range(instances):
		        if (proper_keys[dict_index+j][-5:])=='agent':
			    agent_key=proper_keys[dict_index+j]
			
			    if agent_key in infer_dict.keys():
				if k>infer_dict[agent_key]:
					
				    infer_dict[agent_key]=float(person_action_score)
				  
			    else:	
				infer_dict[agent_key]=float(person_action_score)
		    
			else:
			    obj_score=k
			    obj_bbx_score=np.append(obj_bbx,obj_score)
			    infer_dict[proper_keys[dict_index+j]]=obj_bbx_score.tolist()
		    dict_index+=number_of_roles[index]
	        all_detections.append(infer_dict)




	return all_detections

