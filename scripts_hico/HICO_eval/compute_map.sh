#!/bin/bash
#source /home/iftekhar/virenvs/no2/bin/activate
SUBSET="test"
PRED_HOI_DETS="../../$1/test.pickle"
OUT_DIR="../../$1/"
#PROC_DIR="/home/iftekhar/no_frill/data_symlinks/hico_processed/test"
PROC_DIR="hico_processed/"
num=$2
echo "Arg 2:$num"
python3 -m compute_map --pred_hoi_dets_hdf5 $PRED_HOI_DETS --out_dir $OUT_DIR --proc_dir $PROC_DIR --subset $SUBSET --num_processes $num
#python3 -m comput_map_kn --pred_hoi_dets_hdf5 $PRED_HOI_DETS --out_dir $OUT_DIR --proc_dir $PROC_DIR --subset $SUBSET --num_processes $num

python3 -m sample_complexity_analysis --out_dir $OUT_DIR
#source /home/iftekhar/virenvs/HOI/bin/activate

