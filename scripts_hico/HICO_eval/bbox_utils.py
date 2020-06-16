import numpy as np
#import skimage.draw as skdraw


def add_bbox(img,bbox,color=[0,0,0],fill=False,alpha=1):
    x1,y1,x2,y2 = bbox
    
    # Clockwise starting from top left
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]
    
    if fill:
        coords = skdraw.polygon(r,c,shape=img.shape[0:2])
        skdraw.set_color(img,coords,color,alpha=alpha)
        return

    peri_coords = skdraw.polygon_perimeter(r,c,shape=img.shape[0:2])
    skdraw.set_color(img,peri_coords,color,alpha=alpha)


def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


def compute_iou(bbox1,bbox2,verbose=False):
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 


def compute_area_batch(bbox):
    x1,y1,x2,y2 = [bbox[:,i] for i in range(4)]
    area = np.zeros(x1.shape[0])
    valid_mask = np.logical_and(x2 > x1, y2 > y1)
    area_ = (x2 - x1 + 1) * (y2 - y1 + 1)
    area[valid_mask] = area_[valid_mask]
    return area


def compute_iou_batch(bbox1,bbox2,verbose=False):
    x1,y1,x2,y2 = [bbox1[:,i] for i in range(4)]
    x1_,y1_,x2_,y2_ = [bbox2[:,i] for i in range(4)]
    
    x1_in = np.maximum(x1,x1_)
    y1_in = np.maximum(y1,y1_)
    x2_in = np.minimum(x2,x2_)
    y2_in = np.minimum(y2,y2_)
    
    intersection_bbox = np.stack((x1_in,y1_in,x2_in,y2_in),1)
    intersection = compute_area_batch(bbox=intersection_bbox)
    
    area1 = compute_area_batch(bbox1)
    area2 = compute_area_batch(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    if verbose:
        return iou, intersection, union

    return iou 
    

def vis_bbox(bbox,img,color=(0,0,0),modify=False):
    im_h,im_w = img.shape[0:2]
    x1,y1,x2,y2 = bbox
    x1 = max(0,min(x1,im_w-1))
    x2 = max(x1,min(x2,im_w-1))
    y1 = max(0,min(y1,im_h-1))
    y2 = max(y1,min(y2,im_h-1))
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]

    if modify:
        img_ = img
    else:
        img_ = np.copy(img)

    rr,cc = skdraw.polygon(r,c,img.shape[:2])
    skdraw.set_color(img_,(rr,cc),color,alpha=0.2)

    rr,cc = skdraw.polygon_perimeter(r,c,img.shape[:2])
    for k in range(3):
        img_[rr,cc,k] = color[k]

    return img_


def vis_bboxes(bboxes,img,color=(0,0,0),modify=False):
    if modify:
        img_ = img
    else:
        img_ = np.copy(img)

    for bbox in bboxes:
        img_ = vis_bbox(bbox,img_,color,True)

    return img_


def join_bboxes_by_line(bbox1,bbox2,img,color=(255,0,255),modify=False):
    im_h,im_w = img.shape[0:2]
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2

    c0 = 0.5*(x1+x2)
    r0 = 0.5*(y1+y2)
    c1 = 0.5*(x1_+x2_)
    r1 = 0.5*(y1_+y2_)
    r0,c0,r1,c1 = [int(x) for x in [r0,c0,r1,c1]]
    c0 = max(0,min(c0,im_w-1))
    c1 = max(0,min(c1,im_w-1))
    r0 = max(0,min(r0,im_h-1))
    r1 = max(0,min(r1,im_h-1))
    rr,cc,val = skdraw.draw.line_aa(r0,c0,r1,c1)
    
    if modify:
        img_ = img
    else:
        img_ = np.copy(img)

    for k in range(3):
        img_[rr,cc,k] = val*color[k]

    rr,cc = skdraw.circle(r0,c0,4,img_.shape[:2])
    for k in range(3):
        img_[rr,cc,k] = color[k]

    rr,cc = skdraw.circle(r1,c1,4,img_.shape[:2])
    for k in range(3):
        img_[rr,cc,k] = color[k]

    return img_


def vis_sub_obj_bboxes(
        sub_bboxes,
        obj_bboxes,
        img,
        sub_color=(0,0,255),
        obj_color=(255,0,0),
        modify=False):

    img_ = vis_bboxes(sub_bboxes,img,sub_color,modify)
    img_ = vis_bboxes(obj_bboxes,img_,obj_color,modify=True)
    
    for sub_bbox,obj_bbox in zip(sub_bboxes,obj_bboxes):
        img_ = join_bboxes_by_line(sub_bbox,obj_bbox,img_,modify=True)

    return img_


def vis_human_keypts(
        img,
        keypts,
        radius=2,
        pt_color=(0,255,255),
        line_color=(0,255,255),
        modify=False):
    LINKS = [
        (0,1),
        (1,2),
        (2,3),
        (3,4),
        (1,5),
        (5,6),
        (6,7),
        (0,15),
        (15,17),
        (0,14),
        (14,16),
        (1,8),
        (8,9),
        (9,10),
        (1,11),
        (11,12),
        (12,13),
        (8,11)
    ]

    if modify:
        img_ = img
    else:
        img_ = np.copy(img)

    h,w = img.shape[:2]

    for i,j in LINKS:
        c0,r0,conf0 = keypts[i]
        c1,r1,conf1 = keypts[j]
        r0,r1 = [max(0,min(h-1,int(v))) for v in [r0,r1]]
        c0,c1 = [max(0,min(w-1,int(v))) for v in [c0,c1]]
        if conf0 > 0 and conf1 > 0:
            rr,cc,val = skdraw.draw.line_aa(r0,c0,r1,c1)
            for k in range(3):
                img_[rr,cc,k] = val*line_color[k]

    num_keypts = keypts.shape[0]
    for i in range(num_keypts):
        c,r,conf = keypts[i]
        if conf==0.0:
            continue
        
        rr,cc = skdraw.circle(r,c,radius,img_.shape[:2])
        for k in range(3):
            img_[rr,cc,k] = pt_color[k]

    return img_


