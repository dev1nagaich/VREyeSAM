import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add segment-anything-2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set relative paths from current directory
images_dir = "VRBiomSegM/test/images"
probabilistic_masks_folder = "VRBiomSegM/test/PROBABILISTIC_MASK_PREDICTION"
binary_masks_folder = "VRBiomSegM/test/BINARY_MASK_PREDICTION"


# Create output directories
os.makedirs(probabilistic_masks_folder, exist_ok=True)
os.makedirs(binary_masks_folder, exist_ok=True)

# Model paths - relative to segment-anything-2 directory
sam2_checkpoint = "segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
FINE_TUNED_MODEL_WEIGHTS = "segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

# Load the fine-tuned model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

# Target output mask size
target_size = (400, 300)  # width, height

# Function to read and resize images for input
def read_image(image_path):
    img = cv2.imread(image_path)[..., ::-1]
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img

# Gather image paths
image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

# Process each image
for image_path in image_paths:
    filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename)[0]

    # Load image
    image = read_image(image_path)

    # Generate random input points
    num_samples = 30
    input_points = np.random.randint(0, min(image.shape[:2]), (num_samples, 1, 2))

    # Inference
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    # Convert to numpy
    np_masks = np.array(masks[:, 0]).astype(np.float32)
    np_scores = scores[:, 0]

    # Normalize scores
    score_sum = np.sum(np_scores)
    if score_sum > 0:
        normalized_scores = np_scores / score_sum
    else:
        normalized_scores = np.ones_like(np_scores) / len(np_scores)

    # Generate probabilistic mask
    prob_mask = np.sum(np_masks * normalized_scores[:, None, None], axis=0)
    prob_mask = np.clip(prob_mask, 0, 1)

    # Resize to 400x300 using bilinear interpolation for probabilistic mask
    prob_mask_resized = cv2.resize(prob_mask, target_size, interpolation=cv2.INTER_LINEAR)
    prob_mask_uint8 = (prob_mask_resized * 255).astype(np.uint8)

    # Save probabilistic mask
    prob_save_path = os.path.join(probabilistic_masks_folder, f"{filename_no_ext}.jpg")
    Image.fromarray(prob_mask_uint8).save(prob_save_path)

    # Threshold to get binary mask
    binary_mask = (prob_mask > 0.2).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Save binary mask
    binary_save_path = os.path.join(binary_masks_folder, f"{filename_no_ext}.jpg")
    Image.fromarray(binary_mask_resized).save(binary_save_path)

    print(f"Saved: Probabilistic -> {prob_save_path}, Binary -> {binary_save_path}")
