import os
import argparse

import load as io
# as io
from hico_constants import HicoConstants

parser = argparse.ArgumentParser()
parser.add_argument(
    '--out_dir', 
    type=str, 
    default=None,
    required=True,
    help='Output directory')

def compute_mAP(APs,hoi_ids):
    return sum([APs[hoi_id] for hoi_id in hoi_ids]) / len(hoi_ids)

def main():
    args = parser.parse_args()
    
    data_const = HicoConstants()
    bin_to_hoi_ids = io.load_json_object(data_const.bin_to_hoi_ids_json)
    
    mAP_json = os.path.join(args.out_dir,'mAP.json')
    APs = io.load_json_object(mAP_json)['AP']
    bin_map = {}
    bin_count = {}
    for bin_id,hoi_ids in bin_to_hoi_ids.items():
        bin_map[bin_id] = compute_mAP(APs,hoi_ids)

    non_rare_hoi_ids = []
    for ul in bin_to_hoi_ids.keys():
        if ul=='10':
            continue
        non_rare_hoi_ids += bin_to_hoi_ids[ul]

    sample_complexity_analysis = {
        'bin': bin_map,
        'full': compute_mAP(APs,APs.keys()),
        'rare': bin_map['10'],
        'non_rare': compute_mAP(APs,non_rare_hoi_ids)
    }

    sample_complexity_analysis_json = os.path.join(args.out_dir,'sample_complexity_analysis.json')
    io.dump_json_object(sample_complexity_analysis,sample_complexity_analysis_json)


    bin_names = sorted([int(ul) for ul in bin_map.keys()])
    bin_names = [str(ul) for ul in bin_names]
    bin_headers = ['0'] + bin_names
    bin_headers = [bin_headers[i]+'-'+str(int(ul)-1) for i,ul in enumerate(bin_headers[1:])]
    headers = ['Full','Rare','Non-Rare'] + bin_headers

    sca = sample_complexity_analysis
    values = [sca['full'],sca['rare'],sca['non_rare']] + \
        [bin_map[name] for name in bin_names]
    values = [str(round(v*100,2)) for v in values]

    print('Space delimited values that can be copied to spreadsheet and split by space')
    print(' '.join(headers))
    print(' '.join(values))

if __name__=='__main__':
    main()
