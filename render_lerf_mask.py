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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, reasoning, threshold=0.2):
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
        pred_obj_img_path = os.path.join(pred_obj_path,str(idx))
        pred_mask_map_img_path = os.path.join(pred_mask_map_path,str(idx))
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
        pred_mask_map[:, pred_obj_mask_bool] = pred_mask_map[:, pred_obj_mask_bool] * 0.5 + torch.tensor([[0, 0, 1]], device='cuda').reshape(3, 1) * 0.5
        pred_mask_map[:, ~pred_obj_mask_bool] /= 2
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

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
        if 'figurines' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is a green fruit rich in vitamins;which is a small chair for dolls;which is a device used for taking pictures;which is a decorative object made of porcelain;which is a red fruit rich in vitamins;which is a small chair for dolls;which is a bath toy with a red hat"
                # positive_input = "what is green fruit;what is suitable for people to sit down and is green;what can be used to take pictures and is black;what is like a part of a person;what is red fruit;what is suitable for people to sit down and is red;which is the small yellow rubber duck"
            else:
                positive_input = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat"
        elif 'ramen' in dataset.model_path:
            if args.reasoning:
                positive_input = "which are used for eating Asian cuisine;which is the round food item with a hard shell that is often boiled and served in a bowl;which is the clear liquid container that is often served alongside a meal;which is the type of meat that is often served in a bowl with noodles and broth;which are the long, thin, and wavy food items that are often served in a bowl with broth and meat;which is the container that is often used to serve a hot and savory dish"
                # positive_input = "which one is the chopstic on the side of yellow bowl;what is the round, golden, protein-rich object in the bowl;which one is a transparent cup with water in it;which is the big piece of meat in the bowl;which are long and thin noodles;which is the yellow bowl used to hold noodles"
            else:
                positive_input = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl"
        elif 'teatime' in dataset.model_path:
            if args.reasoning:
                positive_input = "which is a round, red fruit;which is a brown paper bag containing baked goods;which is a white cup with a handle;which are round, brown baked goods on a white dish;which is a white, rectangular piece of paper;which is a white, round dish;which is a white, stuffed animal with a blue lanyard;which is a metal utensil with a handle;which is a large, brown, plush toy;which is a clear glass containing a brown liquid"
                # positive_input = "which is red fruit;which is the brown bag on the side of the plate;which cup is used for coffee;which are the cookies;what can be used to wipe hands;what can be used to hold cookies;which is a cute white doll;which is spoon handle;which is the brown bear doll;which is the drink in the transparent glass"
            else:
                positive_input = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass"
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