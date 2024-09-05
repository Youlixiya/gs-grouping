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



# 1. DEVA anything mask
cd Tracking-Anything-with-DEVA/

if [ "$scale" = "1" ]; then
    img_path="../data/${dataset_name}/images"
else
    img_path="../data/${dataset_name}/images_${scale}"
fi

# colored mask for visualization check
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \


mv ./example/output_gaussian_dataset/${dataset_name}/Annotations ./example/output_gaussian_dataset/${dataset_name}/Annotations_color

# gray mask for training
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  
# 2. copy gray mask to the correponding data path
cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations ../data/${dataset_name}/object_mask
cd ..

CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "../data/ovs3d/lawn/images_4" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/lawn" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7

cp -r ./example/output_gaussian_dataset/lawn/Annotations ../data/ovs3d/lawn/object_mask

CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "../data/messy_rooms/large_corridor_25/images" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/large_corridor_25" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7

cp -r ./example/output_gaussian_dataset/large_corridor_25/Annotations ../data/messy_rooms/large_corridor_25/object_mask

CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "../data/ovs3d/sofa/images_4" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/sofa" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7

cp -r ./example/output_gaussian_dataset/sofa/Annotations ../data/ovs3d/sofa/object_mask

CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "../data/ovs3d/room/images_4" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/room" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7

cp -r ./example/output_gaussian_dataset/room/Annotations ../data/ovs3d/room/object_mask