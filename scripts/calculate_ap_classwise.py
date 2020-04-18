#### This script will calculate AP per classwise without considering bounding boxes precision. This can be used to have a sanity check on the learnt model###

from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import pandas as pd
import torch
import numpy as np
import os
import random
NO_VERBS = 29
VERBS_NO_COCO=80

VERB2ID=['carry', 'catch', 'cut_instr', 'cut_obj', 'drink', 'eat_instr', 'eat_obj', 'hit_instr', 'hit_obj', 'hold', 'jump', 'kick', 'lay', 'look', 'point', 'read', 'ride', 'run', 'sit', 'skateboard', 'ski', 'smile', 'snowboard', 'stand', 'surf', 'talk_on_phone', 'throw', 'walk', 'work_on_computer']


coco_verbs=[u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush']

threshold=0.1
def class_AP(*args):
	result=[]
	predicted_score=args[0]
	true_score=args[1]
	predicted_single_class=args[2]
	true_single_class=args[3]
        mean=0	
	for k in range(NO_VERBS):
		if VERB2ID[k]: 
			predicted=predicted_score[:,k]
			true=true_score[:,k]
			try:
				AP_s=average_precision_score(true,predicted)*100
			except: 
				
                                import pdb;pdb.set_trace()
			
			mean+=AP_s
			result.append((VERB2ID[k],AP_s))

	result.append(('Mean',mean/NO_VERBS))
	mean=0.0
	counter=0
	return result,[('AP',average_precision_score(true_single_class,predicted_single_class)*100)]

if __name__ == "__main__":
	predicted_score=np.random.random_sample([10, 29])
        true_score=np.random.randint(2, size=(10, 29))
	predicted_score_single=np.random.random_sample([10, 1]) 
	true_score_single=np.random.randint(2, size=(10, 1))
    	final=class_AP(predicted_score,true_score,predicted_score_single,true_score_single)
        print(final)
