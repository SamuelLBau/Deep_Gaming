import shutil
import os
from glob import glob
import argparse

src_dir = "./sample_networks/"
dst_dir = "./saved_networks/"
if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)

parser = argparse.ArgumentParser(description='Replace existing networks with sample_networks')
parser.add_argument("--confirm",help="Confirm Replacement of directories",required=False,action="store_true")#run_test = False


args = parser.parse_args()
confirmed = args.confirm

src_list = glob(src_dir+"*")

network_list = [os.path.basename(dir.replace("\\","/")) for dir in src_list]
dst_list     = [dst_dir + network for network in network_list]

conflict_list = []
for dst in dst_list:
    if os.path.isdir(dst):
        conflict_list.append(dst)
        
if len(conflict_list) > 0 and not confirmed:
    print("\n\n\n")
    print("WARNING, this operation will overwrite the following directories:")
    print(conflict_list)
    print("Please use <python load_sample_networks.py --confirm> to confirm deletion and replacement of existing folders")
else:
    print("Copying the following directories:")
    print(src_list)
    for id,[src,dst] in enumerate(zip(src_list,dst_list)):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src,dst)
        


