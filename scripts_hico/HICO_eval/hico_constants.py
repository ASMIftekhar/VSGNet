import os

import load as io


class HicoConstants(io.JsonSerializableClass):
    def __init__(
            self,
            clean_dir=os.path.join(os.getcwd(),'data_symlinks/hico_clean'),
            #proc_dir=os.path.join(os.getcwd(),'data_symlinks/hico_processed')):
            proc_dir=os.path.join(os.getcwd(),'hico_processed/')):
        self.clean_dir = clean_dir
        self.proc_dir = proc_dir

        # Clean constants
        self.anno_bbox_mat = os.path.join(self.clean_dir,'anno_bbox.mat')
        self.anno_mat = os.path.join(self.clean_dir,'anno.mat')
        self.hico_list_hoi_txt = os.path.join(
            self.clean_dir,
            'hico_list_hoi.txt')
        self.hico_list_obj_txt = os.path.join(
            self.clean_dir,
            'hico_list_obj.txt')
        self.hico_list_vb_txt = os.path.join(
            self.clean_dir,
            'hico_list_vb.txt')
        self.images_dir = os.path.join(self.clean_dir,'images')

        # Processed constants
        self.anno_list_json = os.path.join(self.proc_dir,'anno_list.json')
        self.hoi_list_json = os.path.join(self.proc_dir,'hoi_list.json')
        self.object_list_json = os.path.join(self.proc_dir,'object_list.json')
        self.verb_list_json = os.path.join(self.proc_dir,'verb_list.json')

        # Need to run split_ids.py
        self.split_ids_json = os.path.join(self.proc_dir,'split_ids.json')

        # Need to run hoi_cls_count.py
        self.hoi_cls_count_json = os.path.join(self.proc_dir,'hoi_cls_count.json')
        self.bin_to_hoi_ids_json = os.path.join(self.proc_dir,'bin_to_hoi_ids.json')

        self.faster_rcnn_boxes = os.path.join(self.proc_dir,'faster_rcnn_boxes')
