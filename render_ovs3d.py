# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2

from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor

from render import feature_to_rgb, visualize_obj


# data/ovs3d/bed/segmentations/00/banana.png


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, reasoning, threshold=0.1):
    if reasoning:
        reasoning_path = 'reasoning'
    else:
        reasoning_path = ''
    render_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), reasoning_path, "renders")
    gts_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), reasoning_path, "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), reasoning_path, "objects_feature16")
    pred_obj_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), reasoning_path, "test_mask")
    pred_mask_map_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), reasoning_path,"test_mask_map")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_mask_map_path, exist_ok=True)

    # Use Grounded-SAM on the first frame
    results0 = render(views[0], gaussians, pipeline, background)
    rendering0 = results0["render"]
    rendering_obj0 = results0["render_object"]
    logits = classifier(rendering_obj0)
    pred_obj = torch.argmax(logits,dim=0)

    image = (rendering0.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
    text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
    Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8],'grounded-sam---'+TEXT_PROMPT+'.png'))
    selected_obj_ids = select_obj_ioa(pred_obj, text_mask)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pred_obj_img_path = os.path.join(pred_obj_path,view.image_name)
        pred_mask_map_img_path = os.path.join(pred_mask_map_path,view.image_name)
        makedirs(pred_obj_img_path, exist_ok=True)
        makedirs(pred_mask_map_img_path, exist_ok=True)
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)

        if len(selected_obj_ids) > 0:
            prob = torch.softmax(logits,dim=0)

            pred_obj_mask = prob[selected_obj_ids, :, :] > threshold
            pred_obj_mask_bool = pred_obj_mask.any(dim=0)
            pred_obj_mask = (pred_obj_mask_bool.squeeze().cpu().numpy() * 255).astype(np.uint8)
        else:
            pred_obj_mask_bool = torch.zeros_like(view.objects, dtype=torch.bool)
            pred_obj_mask = torch.zeros_like(view.objects).cpu().numpy()
            
        pred_mask_map = rendering.clone()
        pred_mask_map[:, pred_obj_mask_bool] = pred_mask_map[:, pred_obj_mask_bool] * 0.5 + torch.tensor([[1, 0, 0]], device='cuda').reshape(3, 1) * 0.5
        pred_mask_map[:, ~pred_obj_mask_bool] /= 2
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
        print(rendering.shape)

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        print(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(pred_mask_map, os.path.join(pred_mask_map_img_path, TEXT_PROMPT + ".png"))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        args.reasoning
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        # sam-hq
        sam_checkpoint = 'ckpts/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        # Text prompt
        if 'bed' in dataset.model_path:
            if args.reasoning:
                positive_input ="which is a fruit with a yellow peel;which is an object that can be worn on the feet;which is a device used for taking pictures;which is a part of the human body;which is red,leathern object used to put items in;which is a piece of fabric used for covering a bed"
                # positive_input = "which is a yellow fruit often eaten as a snack;which is a black shoe with a gold buckle;which is a device used for taking pictures;which is a part of the human body used for holding objects;which is a red bag with a quilted pattern;which is a white sheet with black lines"
                # positive_input = "which is the yellow fruit;which can be worn on the foot;which can be used to take photos;which is the part of person, excluding other objects;which is red and leather;where is a good place to lie down"
            else:
                positive_input = "banana;black leather shoe;camera;hand;red bag;white sheet"
        elif 'bench' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is an object used for dressing up;which is a fruit that is green;which is a small vehicle used for off-road driving;which is an animal that is orange;which is a wall made of pebbled concrete;which is a dessert that is a Portuguese egg tart;which is an object made of wood"
                # positive_input = "which is a toy used for dressing up;which is a green fruit that grows in clusters;which is a small toy car designed for off-road use;which is a feline with orange fur;which is a wall made of concrete with embedded pebbles;which is a pastry with a custard filling;which is a material used for building and furniture"
                # positive_input = "which is a cute humanoid doll that girls like;which is green fruit;which one is the model of the vehicle;which is an animal;which is made of many stones;which is like baked food;which is made of wood"
            else:
                positive_input = "dressing doll;green grape;mini offroad car;orange cat;pebbled concrete wall;Portuguese egg tart;wood"
        elif 'lawn' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is a red fruit rich in vitamins;which is a cap with a sports team logo;which is a device used for fastening paper;which is a black device used for listening to audio;which is a liquid used for cleaning hands;which is a green grassy area"
                # positive_input = "which is the red fruit;which is worn on the head and is white;which is small device used for stapling paper;which can convert electric signals into sounds;which is bottled;which is an area of ground covered in short grass"
            else:
                positive_input = "red apple;New York Yankees cap;stapler;black headphone;hand soap;green lawn"
        elif 'room' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is a type of material used for furniture and construction;which is a toy that makes a loud noise when squeezed;which is a container made from woven materials;which is a small, furry animal with long ears;which is a prehistoric creature that lived millions of years ago;which is a round, white ball used in a sport"
                # positive_input = "which is background wood board;which is a yellow animal doll;which can be uesd to hold a water bottle;which is a cute mammal doll;which has a long tail;which is spherical and white"
            else:
                positive_input = "wood;shrilling chicken;weaving basket;rabbit;dinosaur;baseball"
        elif 'sofa' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is a yellow electric-type creature;which is a deck of playing cards;which is a piece of furniture;which is a handheld gaming device;which is a model of a robot;which is a device used to play video games"
                # positive_input = "which is a yellow plush toy with a hat;which is a deck of cards with a colorful design;which is a piece of furniture with a soft, grey surface;which is a red handheld gaming device;which is a blue and white action figure;which is a white gaming controller with buttons and joysticks"
                # positive_input = "which is the yellow doll;what is made of cards stacked together;where can I sit down;which is red and looks like a controller;which is the body of a robot model;which can be used to play games and is large and white"
            else:
                positive_input = "Pikachu;a stack of UNO cards;grey sofa;a red Nintendo Switch joy-con controller;Gundam;Xbox wireless controller"                   
        else:
            raise NotImplementedError   # You can provide your text prompt here

        positives = positive_input.split(";")
        print("Text prompts:    ", positives)

        for TEXT_PROMPT in positives:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, args.reasoning)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, args.reasoning)


             

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)