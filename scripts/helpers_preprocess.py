import json
import numpy as np
import argparse
from random import randint
import cv2

with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)    


ANN_FILE_train = all_data_dir+'Annotations_vcoco/train_annotations.json' 
ANN_FILE_val=all_data_dir+'Annotations_vcoco/val_annotations.json'
ANN_FILE_test=all_data_dir+'Annotations_vcoco/test_annotations.json'
with open(ANN_FILE_train) as fp:
    ANNOTATIONS_train = json.load(fp)
with open(ANN_FILE_test) as fp:
    ANNOTATIONS_test = json.load(fp)
with open(ANN_FILE_val) as fp:
    ANNOTATIONS_val = json.load(fp)

    

    
OBJ_PATH_train_s = all_data_dir+'Object_Detections_vcoco/train/'
OBJ_PATH_test_s= all_data_dir+'Object_Detections_vcoco/val/' 


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

MATCHING_IOU = .5
NO_VERBS = 29




def get_detections(segment_key,flag):
    if flag=='train':
	annotation = ANNOTATIONS_train[str(segment_key)]
	cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (segment_key)
	SCORE_TH = 0.6
	SCORE_OBJ=0.3

	select_threshold=2000000
    elif flag=='test':
	annotation = ANNOTATIONS_test[str(segment_key)]
	cur_obj_path_s = OBJ_PATH_test_s + "COCO_val2014_%.12i.json" % (segment_key)
	SCORE_TH = 0.6
	SCORE_OBJ=0.3
	select_threshold=2000000
    elif flag=='val':
	 annotation = ANNOTATIONS_val[str(segment_key)]
         cur_obj_path_s = OBJ_PATH_train_s + "COCO_train2014_%.12i.json" % (segment_key)
	 SCORE_TH = 0.6
	 SCORE_OBJ=0.3
	 select_threshold=2000000

    annotation = clean_up_annotation(annotation)
    with open(cur_obj_path_s) as fp:detections = json.load(fp)
    
    img_H = detections['H']
    img_W = detections['W']
    shape=[img_W,img_H] 
    persons_d, objects_d = analyze_detections(detections,SCORE_TH,SCORE_OBJ)
    d_p_boxes,scores_persons,class_id_humans = get_boxes_det(persons_d, img_H, img_W)
    d_o_boxes,scores_objects,class_id_objects = get_boxes_det(objects_d, img_H, img_W)
    if len(d_p_boxes)>select_threshold:
        d_p_boxes,scores_persons,class_id_humans= d_p_boxes[0:select_threshold],scores_persons[0:select_threshold],class_id_humans[0:select_threshold]
    if len(d_o_boxes)>select_threshold-1:
        d_o_boxes,scores_objects,class_id_objects= d_o_boxes[0:select_threshold-1],scores_objects[0:select_threshold-1],class_id_objects[0:select_threshold-1]
    scores_objects.insert(0,1)
    return d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape



    
    
def get_compact_detections(segment_key,flag):    
    d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape=get_detections(segment_key,flag)
    img_W=shape[0]
    img_H=shape[1]
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    persons_np = np.zeros([no_person_dets, 4], np.float32)
    objects_np = np.zeros([no_object_dets+1, 4], np.float32)
    class_id_objects.insert(0,1)
    if no_person_dets != 0:
	persons_np = np.array(d_p_boxes, np.float32)
 
		
    objects_np = np.array([[0,0,0,0]] + d_o_boxes, np.float32) 
    persons_np=persons_np/ np.array([img_W, img_H, img_W, img_H])
    objects_np=objects_np/ np.array([img_W, img_H, img_W, img_H])

    return { 'person_bbx':persons_np, 'objects_bbx':objects_np,
	    'person_bbx_score':scores_persons,'objects_bbx_score':scores_objects,
            'class_id_objects':class_id_objects}
    
    
    
    
def get_attention_maps(segment_key,flag): 
    compact_detections=get_compact_detections(segment_key,flag)
    persons_np,objects_np=compact_detections['person_bbx'],compact_detections['objects_bbx']
    union_box=[]
    no_person_dets = len(persons_np)
    no_object_dets = len(objects_np)
    for dd_i in range(no_person_dets):
        for do_i in range(len(objects_np)):
            union_box.append(union_BOX(persons_np[dd_i],objects_np[do_i],segment_key))
    return np.concatenate(union_box)


    
    
    
def get_compact_label(segment_key,flag):    
  
    d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape=get_detections(segment_key,flag)
    
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    labels_np = np.zeros([no_person_dets, no_object_dets+1, NO_VERBS], np.int32) 
    
    a_p_boxes = [ann['person_box'] for ann in annotation]
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)

    if no_person_dets != 0 and len(a_p_boxes)!=0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)
        index_for_each_det  = np.argmax(iou_mtx, axis=0)
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_for_each_det[dd]
            if cur_max_iou < MATCHING_IOU: 
                continue
            matched_ann = annotation[index_for_each_det[dd]]
            hoi_anns = matched_ann['hois']
            #Verbs with no actions####
            noobject_hois = [oi for oi in hoi_anns if len(oi['obj_box']) == 0]
            
            for no_hoi in noobject_hois:
                verb_idx = VERB2ID[no_hoi['verb']]
                labels_np[dd, 0, verb_idx] = 1

            # verbs with actions######
            object_hois = [oi for oi in hoi_anns if len(oi['obj_box']) != 0]
             
            a_o_boxes = [oi['obj_box'] for oi in object_hois]
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)


            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                  for ao in range(len(a_o_boxes)):
                    cur_iou = iou_mtx_o[ao,do]
                    # enough iou
                    if cur_iou < MATCHING_IOU:
                        continue
                    current_hoi = object_hois[ao]
                    verb_idx = VERB2ID[current_hoi['verb']]
                    labels_np[dd, do+1, verb_idx] = 1 # +1 because 0 is no object
    

        comp_labels=labels_np.reshape(no_person_dets*(no_object_dets+1),NO_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)
        return{'labels_all':labels_np,'labels_single':labels_single}
    else:
        comp_labels=labels_np.reshape(no_person_dets*(no_object_dets+1),NO_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)
        return{'labels_all':labels_np,'labels_single':labels_single}
    
def get_bad_detections(segment_key,flag): #Detections Without any person#    
    
    labels_all=get_compact_label(segment_key,flag)['labels_all']
    if labels_all.size==0:
        return True
    else:
        return False

        






def union_BOX(roi_pers,roi_objs,segment_key,H=64,W=64):
        assert H==W
        roi_pers=np.array(roi_pers*H,dtype=int)
        roi_objs=np.array(roi_objs*H,dtype=int)
        sample_box=np.zeros([1,2,H,W])
        sample_box[0,0,roi_pers[1]:roi_pers[3]+1,roi_pers[0]:roi_pers[2]+1]=100
        sample_box[0,1,roi_objs[1]:roi_objs[3]+1,roi_objs[0]:roi_objs[2]+1]=100
	return sample_box            

	
	

def clean_up_annotation(annotation):
    persons_dict = {}
    for hoi in annotation:
	
    	 
        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
        dkey = tuple(box)
        objects = hoi['object']
        if len(objects['obj_bbx']) == 0: # no obj case
            cur_oi = {  'verb': hoi['Verbs'], 
                        'obj_box':[],
                        #'obj_str': '',
                        }
        else:
            cur_oi = {  'verb': hoi['Verbs'], 
                        'obj_box':[int(coord) for coord in hoi['object']['obj_bbx']],
                        #'obj_str': hoi['object']['obj_name'],
                        }
            
        if dkey in persons_dict:
            persons_dict[dkey]['hois'].append(cur_oi)
        else:
            persons_dict[dkey] = {'person_box': box, 'hois': [cur_oi]}
    

    pers_list = []
    for dkey in persons_dict:
        pers_list.append(persons_dict[dkey])

    return pers_list



def get_boxes_det(dets, img_H, img_W): 
    boxes = []
    scores=[]
    class_no=[]
    for det in dets:
        top,left,bottom,right = det['box_coords']
	scores.append(det['score'])
	class_no.append(det['class_no'])
        left, top, right, bottom = left*img_W, top*img_H, right*img_W, bottom*img_H
        #left, top, right, bottom = left, top, right, bottom
        boxes.append([left,top,right,bottom])
    return boxes,scores,class_no
        

def get_iou_mtx(anns, dets):
    no_gt = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_gt, no_dt])

    for gg in range(no_gt):
        gt_box = anns[gg]
        for dd in range(no_dt):
            dt_box = dets[dd]
            iou_mtx[gg,dd] = IoU_box(gt_box,dt_box)

    return iou_mtx



def analyze_detections(detections,SCORE_TH,SCORE_OBJ):
    persons = []
    objects = []
    
    for det in detections['detections']:
        if det['class_str'] == 'person':
            if det['score'] < SCORE_TH:
            	continue
            persons.append(det)
	
        else:
            if det['score'] < SCORE_OBJ:
            	continue
            objects.append(det)

    return persons, objects


def IoU_box(box1, box2):
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    returns intersection over union
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    left_int = max(left1, left2)
    top_int = max(top1, top2)
 
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
 
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
 
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
     
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU

def dry_run():
	ALL_SEGS_train =ANNOTATIONS_train.keys()
	ALL_SEGS_val =ANNOTATIONS_val.keys()
	ALL_SEGS_test =ANNOTATIONS_test.keys()
	ALL_SEGS_train = [int(v) for v in ALL_SEGS_train]
    	ALL_SEGS_train.sort() 
	ALL_SEGS_val = [int(v) for v in ALL_SEGS_val]
    	ALL_SEGS_val.sort()
	new_anns = {} 
	ALL_SEGS_test = [int(v) for v in ALL_SEGS_test]
    	ALL_SEGS_test.sort() 
        bad_detections_train=[]
        bad_detections_val=[]
        bad_detections_test=[]
	print("Doing a test run to detect bad detections\n")
	
	for segkey in (ALL_SEGS_train):

            if get_bad_detections(segkey,"train"):
                 bad_detections_train.append(segkey)
        print("In training set object detector failed to detect any person in the following images:\n{}".format(bad_detections_train))
	
        
	for segkey in (ALL_SEGS_val):
            if get_bad_detections(segkey,"val"):
                 bad_detections_val.append(segkey)
        print("In validation set object detector failed to detect any person in the following images:\n{}".format(bad_detections_val))
	for segkey in (ALL_SEGS_test):
            if get_bad_detections(segkey,"test"):
                 bad_detections_test.append(segkey)
        print("In testing set object detector failed to detect any person in the following images:\n{}".format(bad_detections_test))
        return bad_detections_train,bad_detections_val,bad_detections_test



if __name__ == "__main__":
    new_anns = {}
    compact_dets={}
    att_maps={}
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--type_of_data',type=str,required=False,default='train',help="type_of_data")
    args=parser.parse_args()
    flag=args.type_of_data
    b_d_tr,b_d_val,b_d_test=dry_run()
    phases=['train','val','test']
    from tqdm import tqdm
    for flag in phases:
	    if flag=='train':
		ALL_SEGS =ANNOTATIONS_train.keys()

	    elif flag=='test':
		ALL_SEGS =ANNOTATIONS_test.keys()

	    elif flag=='val':
		ALL_SEGS =ANNOTATIONS_val.keys()
	    ALL_SEGS = [int(v) for v in ALL_SEGS]
	    ALL_SEGS.sort()
	    for segkey in tqdm(ALL_SEGS):
	        if segkey not in (b_d_tr+b_d_val+b_d_test):	
		    new_anns[segkey] = get_compact_label(segkey,flag)
		    compact_dets[segkey] = get_compact_detections(segkey,flag)
		    att_maps[segkey] = get_attention_maps(segkey,flag)
                    import pdb;pdb.set_trace()
		
    pass
