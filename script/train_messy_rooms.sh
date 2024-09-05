#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi


dataset_name="$1"
scale="$2"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training
# python train.py    -s $dataset_folder -r ${scale}  -m output/${dataset_name} --config_file config/gaussian_dataset/train.json --train_split

# CUDA_VISIBLE_DEVICES=1 python train.py  -s data/ovs3d/bed --images images_4  -m output/ovs3d/bed --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python render.py -m output/${dataset_name} --num_classes 256 --images images

# CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -m output/ovs3d/bed --num_classes 256 --skip_train
# CUDA_VISIBLE_DEVICES=1 python render_ovs3d.py -m output/ovs3d/bed --num_classes 256 --reasoning --skip_train

# CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -m output/ovs3d/bench --num_classes 256 --skip_train
# CUDA_VISIBLE_DEVICES=2 python render_ovs3d.py -m output/ovs3d/bench --num_classes 256 --reasoning --skip_train

# CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -m output/ovs3d/lawn --num_classes 256 --skip_train
# CUDA_VISIBLE_DEVICES=3 python render_ovs3d.py -m output/ovs3d/lawn --num_classes 256 --reasoning --skip_train


CUDA_VISIBLE_DEVICES=1 python train.py  -s data/messy_rooms/large_corridor_25 -m output/messy_rooms/large_corridor_25 --config_file config/gaussian_dataset/train.json
CUDA_VISIBLE_DEVICES=1 python render_messy_rooms.py -m output/messy_rooms/large_corridor_25 --num_classes 256 --skip_train