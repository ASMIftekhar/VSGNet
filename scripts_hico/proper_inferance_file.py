import json
import torch
import numpy as np


with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)  
OBJ_PATH_train_s = all_data_dir+'Object_Detections_hico/train/'
OBJ_PATH_test_s= all_data_dir+'Object_Detections_hico/test/'



with open(all_data_dir+'hico_infos/hico_list_hoi.json') as fp:
        hoi=json.load(fp)
with open(all_data_dir+'hico_infos/hico_list_vb.json') as fp:
        list_vbs= json.load(fp)
with open(all_data_dir+'hico_infos/hico_list_obj.json') as fp:
        list_obj = json.load(fp)

with open(all_data_dir+'hico_infos/index.json') as fp:indexes=json.load(fp)


def infer_format(image_id,all_scores_batch,flag,all_detections,pairs_info):
        #flat=com_labels.flatten()
#       print(person_bbxn)i
        this_batch_start=0
        for batch in range(len(image_id)):
    		new_dict = {new_list: [] for new_list in range(600)} 	
		start_end_dict={}
		box={}
		start_end=np.zeros([600,2])
		box_temp=np.zeros([1,9])
                this_image=int(image_id[batch])
                #a=labels.get_compact_labels(this_image,flag)
                persons=all_scores_batch[this_image,'pers_bbx']

                objects=all_scores_batch[this_image,'obj_bbx']
		if flag=='train':
                        cur_obj_path_s = OBJ_PATH_train_s + "HICO_train2015_%.8i.json" % (this_image)
		elif flag=='test':

                        cur_obj_path_s = OBJ_PATH_test_s + "HICO_test2015_%.8i.json" % (this_image)
                with open(cur_obj_path_s) as fp:detections = json.load(fp)
		img_H = detections['H']
                img_W = detections['W']
		persons=np.array([img_W,img_H,img_W,img_H],dtype=float)*persons
		objects=np.array([img_W,img_H,img_W,img_H],dtype=float)*objects
		persons=persons.astype('int32')
		objects=objects.astype('int32')
                class_ids=all_scores_batch[this_image,'class_ids']
		class_ids=class_ids.reshape(len(class_ids),).astype('int32')
                hum_scores=0
                this_batch_pers=int(pairs_info[batch][0])
                this_batch_objs=int(pairs_info[batch][1])
                increment=this_batch_pers*this_batch_objs
  		all_scores=all_scores_batch[this_image,'score']
		#import pdb;pdb.set_trace()
		for ind,cl_id in enumerate(class_ids):
			
			index_verb,index_hoi=indexes[str(cl_id)]['index_verb'],indexes[str(cl_id)]['index_hoi']
			for verb_no,verb in enumerate(index_verb):
				try:
					tmp_score=np.array([all_scores[ind,verb]])
				except:	
					import pdb;pdb.set_trace()
				tmp_box=np.concatenate([persons[ind],objects[ind],tmp_score])
				new_dict[index_hoi[verb_no]].append(tmp_box)

		full_keys = [k for k, v in new_dict.items() if len(v)!=0]
		count=0
		for l in full_keys:
			#import pdb;pdb.set_trace()
			box_temp=np.concatenate([box_temp,new_dict[l]])	
			start_end[l,0]=int(count)
			start_end[l,1]=int(count+len(new_dict[l]))
			count+=len(new_dict[l])
			
		

		start_end_dict['start_end_ids']=start_end
		glo_id="HICO_test2015_%.8i" % (this_image)

		all_detections[glo_id]={'start_end_ids':start_end,'human_obj_boxes_scores':box_temp[1:]}








