import os
import numpy as np
from PIL import Image
import cv2
import sys

dataset_name = sys.argv[1]


gt_folder_path = os.path.join('data','ovs3d', dataset_name,'segmentations')
# You can change pred_folder_path to your output
pred_folder_path = os.path.join('output','ovs3d', dataset_name, 'test/ours_10000_text/reasoning/test_mask')


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt>128).astype('uint8')
    gt = (gt>128).astype('uint8')
    

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def load_mask(mask_path):
    """Load the mask from the given path."""
    if os.path.exists(mask_path):
        return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
    return None

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

iou_scores = {}  # Store IoU scores for each class
biou_scores = {}
class_counts = {}  # Count the number of times each class appears

# prompt_dict
if dataset_name == 'bed':
    prompt_dict = {'banana':'which is a fruit with a yellow peel',
  'black leather shoe': 'which is an object that can be worn on the feet',
  'camera': 'which is a device used for taking pictures',
  'hand': 'which is a part of the human body',
  'red bag': 'which is red,leathern object used to put items in',
  'white sheet': 'which is a piece of fabric used for covering a bed'}
    # prompt_dict = {"banana":"which is a yellow fruit often eaten as a snack","black leather shoe":"which is a black shoe with a gold buckle","camera":"which is a device used for taking pictures","hand":"which is a part of the human body used for holding objects","red bag":"which is a red bag with a quilted pattern","white sheet":"which is a white sheet with black lines"}
    # prompt_dict = {"banana":"which is the yellow fruit" ,"black leather shoe":"which can be worn on the foot","camera":"which can be used to take photos","hand":"which is the part of person, excluding other objects","red bag":"which is red and leather","white sheet":"where is a good place to lie down"}
#     # index_dict = {"00":"0","04":"1","10":"2","23":"3","30":"4"}
elif dataset_name == 'bench':
    prompt_dict = {"dressing doll": "which is an object used for dressing up",
  "green grape": "which is a fruit that is green",
  "mini offroad car": "which is a small vehicle used for off-road driving",
  "orange cat": "which is an animal that is orange",
  "pebbled concrete wall": "which is a wall made of pebbled concrete",
  "Portuguese egg tart": "which is a dessert that is a Portuguese egg tart",
  "wood": "which is the object made of wood"}
    # prompt_dict = {"dressing doll":"which is a toy used for dressing up","green grape":"which is a green fruit that grows in clusters","mini offroad car":"which is a small toy car designed for off-road use","orange cat":"which is a feline with orange fur","pebbled concrete wall":"which is a wall made of concrete with embedded pebbles","Portuguese egg tart":"which is a pastry with a custard filling","wood":"which is a material used for building and furniture"}
    # prompt_dict = {"dressing doll":"which is a cute humanoid doll that girls like","green grape":"which is green fruit", "mini offroad car":"which one is the model of the vehicle","orange cat":"which is an animal","pebbled concrete wall":"which is made of many stones", "Portuguese egg tart":"which is like baked food","wood":"which is made of wood"}
#     index_dict = {"02":"0","05":"4","25":"3","27":"1","32":"2"}
elif dataset_name == 'sofa':
    prompt_dict = {'Pikachu': 'which is a yellow electric-type creature',
  'a stack of UNO cards': 'which is a deck of playing cards',
  'grey sofa': 'which is a piece of furniture',
  'a red Nintendo Switch joy-con controller': 'which is a handheld gaming device',
  'Gundam': 'which is a model of a robot',
  'Xbox wireless controller': 'which is a device used to play video games'}
    # prompt_dict = {"Pikachu":"which is a yellow plush toy with a hat","a stack of UNO cards":"which is a deck of cards with a colorful design","grey sofa":"which is a piece of furniture with a soft, grey surface","a red Nintendo Switch joy-con controller":"which is a red handheld gaming device","Gundam":"which is a blue and white action figure","Xbox wireless controller":"which is a white gaming controller with buttons and joysticks"}
    # prompt_dict = {"Pikachu":"which is the yellow doll","a stack of UNO cards":"what is made of cards stacked together", "grey sofa":"where can I sit down","a red Nintendo Switch joy-con controller":"which is red and looks like a controller","Gundam":"which is the body of a robot model","Xbox wireless controller":"which can be used to play games and is large and white"}
# 
elif dataset_name == 'room':
    prompt_dict = {"wood":"which is a type of material used for furniture and construction","shrilling chicken":"which is a toy that makes a loud noise when squeezed","weaving basket":"which is a container made from woven materials","rabbit":"which is a small, furry animal with long ears","dinosaur":"which is a prehistoric creature that lived millions of years ago","baseball":"which is a round, white ball used in a sport"}
    # prompt_dict = {"wood":"which is background wood board", "shrilling chicken":"which is a yellow animal doll", "weaving basket":"which can be uesd to hold a water bottle","rabbit":"which is a cute mammal doll","dinosaur":"which has a long tail","baseball":"which is spherical and white"}
#                          
elif dataset_name == 'lawn':
    prompt_dict = {"red apple":"which is a red fruit rich in vitamins","New York Yankees cap":"which is a cap with a sports team logo","stapler":"which is a device used for fastening paper","black headphone":"which is a black device used for listening to audio","hand soap":"which is a liquid used for cleaning hands","green lawn":"which is a green grassy area"}
    # prompt_dict = {"red apple":"which is the red fruit","New York Yankees cap":"which is worn on the head and is white","stapler":"which is small device used for stapling paper","black headphone":"which can convert electric signals into sounds","hand soap":"which is bottled", "green lawn":"which is an area of ground covered in short grass"}
#     index_dict = {"01":"2","03":"1","09":"0","13":"4","29":"3"}

elif dataset_name == 'large_corridor_25':
    prompt_dict = {"green pot": "which is an object used for planting",
     "white plush doll": "which is an object used for comfort",
      "white bottle": "which is an white bottle-shaped object",
     "brown bowl": 'which is a brown object that can be used for eating', 
     "blue lunch box": 'which is a blue object with dinosaur print'}
elif dataset_name == 'large_corridor_50':
    prompt_dict = {'colorful clock':'which is a colorful object used for telling time',
    'black and white shoe':'which is a black and white object that can be worn on the feet',
    'blue bag':'which is a blue object used for carrying items with balls patterns on it',
    'wooden box':'which is a brown object used for storing and organizing items',
    'black cable':'which is a black object used for connecting electronic devices'}
elif dataset_name == 'large_corridor_100':
    prompt_dict = {"white and black shoe":"which is a white object that can be worn on the feet",
    "blue and yellow sandal":"which is a blue and yellow object that can be worn on the feet",
    "blue and white striped towel":"which is a blue and white object that can be used for wiping",
    "green plastic box":"which is a green object with holes that can be used for storage",
    "red and blue doll":"which is a red and blue object appearing in games"}


# Iterate over each image and category in the GT dataset
for image_name in os.listdir(gt_folder_path):
    gt_image_path = os.path.join(gt_folder_path, image_name)
    # pred_image_path = os.path.join(pred_folder_path, image_name)
    
    if os.path.isdir(gt_image_path):
        for cat_file in os.listdir(gt_image_path):
            cat_id = cat_file.split('.')[0]  # Assuming cat_file format is "cat_id.png"
            gt_mask_path = os.path.join(gt_image_path, cat_file)
            pred_mask_path = os.path.join(pred_folder_path, image_name, prompt_dict[cat_id]+'.png')
            
#prompt_dict_teatime[cat_id]+'.png'


            gt_mask = load_mask(gt_mask_path)
            pred_mask = load_mask(pred_mask_path)
            print("GT:  ",gt_mask_path)
            print("Pred:  ",pred_mask_path)

            if gt_mask is not None and pred_mask is not None:
                # Resize prediction mask to match GT mask shape if they are different
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = resize_mask(pred_mask, gt_mask.shape)

                iou = calculate_iou(gt_mask, pred_mask)
                biou = boundary_iou(gt_mask, pred_mask)
                print(iou)
                print("IoU: ",iou," BIoU:   ",biou)
                if cat_id not in iou_scores:
                    iou_scores[cat_id] = []
                    biou_scores[cat_id] = []
                iou_scores[cat_id].append(iou)
                biou_scores[cat_id].append(biou)
                class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

# Calculate mean IoU for each class
mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

# Calculate overall mean IoU
overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

print("Mean IoU per class:", mean_iou_per_class)
print("Mean Boundary IoU per class:", mean_biou_per_class)
print("Overall Mean IoU:", overall_mean_iou)
print("Overall Boundary Mean IoU:", overall_mean_biou)