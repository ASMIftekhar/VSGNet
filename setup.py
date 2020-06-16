import argparse
import json



parser=argparse.ArgumentParser()
parser.add_argument('-d','--Directory',type=str,required=False,default='../All_data/',help='Directory where all folders are kept')
args=parser.parse_args()

dirr=args.Directory

with open('infos/directory.json','w') as fp:json.dump(dirr,fp)
