import os
import sys
import cv2
import torch
import torch.nn.utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add segment-anything-2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Paths
checkpoint_path = "segment-anything-2/checkpoints/VREyeSAM_uncertainity_best.torch"  # your saved model
FINE_TUNED_MODEL_NAME = "VREyeSAM_uncertainity"
data_dir = "VRBiomSegM/train"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")

# Load dataset
train_df = pd.read_csv(os.path.join(data_dir, "pretrained_model/train_combined.csv"))
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_data = [{"image": os.path.join(images_dir, row['ImageId']), "annotation": os.path.join(masks_dir, row['MaskId'])} for _, row in train_df.iterrows()]

def read_batch(data):
    """
    Loads and preprocesses a random training sample with data augmentation.
    Returns: image, binary mask, random point prompt, and number of valid masks

    """

    ent = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[..., ::-1]
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    if Img is None or ann_map is None:
        return None, None, None, 0
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    binary_mask = np.where(ann_map > 0, 1, 0).astype(np.uint8)
    coords = np.argwhere(binary_mask > 0)
    points = np.array([[yx[1], yx[0]] for yx in coords[np.random.choice(len(coords), min(len(coords), 1))]])
    return Img, binary_mask[np.newaxis, ...], points[np.newaxis, ...], 1

# Load SAM2 Model

# Resume from checkpoint if available, otherwise start fresh training
# This allows continuing training from a previously saved model state

# Use absolute path for config to avoid Hydra search path issues
config_dir = os.path.dirname(__file__)
model_cfg = os.path.join(config_dir, "segment-anything-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")
sam2_checkpoint = "segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Resume from checkpoint
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)
    predictor.model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting fresh.")


# Main training loop with uncertainty-weighted loss and gradient accumulation

predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=0.0001, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=87400, gamma=0.2)
accumulation_steps = 4

NO_OF_STEPS = 437000  # number of steps to continue training
loss_values = []
steps_recorded = []

# For saving best model
best_iou = 0.0
best_model_path = f"{FINE_TUNED_MODEL_NAME}_best.torch"

for step in range(1, NO_OF_STEPS + 1):
    with torch.cuda.amp.autocast():
        image, mask, input_point, num_masks = read_batch(train_data)
        if image is None or mask is None or num_masks == 0:
            continue

        input_label = np.ones((num_masks, 1))
        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label)

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )
        
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        # Prepare ground truth and predictions for loss calculation

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])

        # Calculate uncertainty map using entropy of predicted probabilities
        # Higher uncertainty in ambiguous regions will weight the loss more

        uncertainty_map = -prd_mask * torch.log(prd_mask + 1e-6) - (1 - prd_mask) * torch.log(1 - prd_mask + 1e-6)
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)

        # Uncertainty-weighted binary cross-entropy loss - main contribution
        # Focuses learning on uncertain/difficult regions

        seg_loss = (uncertainty_map * (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6))).mean()
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)

        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)# Calculate IoU between ground truth and predictions. Used for both model evaluation and score loss
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean() # Score loss: align predicted confidence with actual IoU performance

        loss = (seg_loss + score_loss * 0.05) / accumulation_steps # Combined loss: segmentation + score alignment, scaled for gradient accumulation
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()

        scheduler.step()
        loss_values.append(loss.item())
        steps_recorded.append(step)

        # Track and save best performing model based on IoU

        mean_iou = iou.mean().item()
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(predictor.model.state_dict(), best_model_path)
            print(f"Best model saved at step {step} with IoU: {best_iou:.4f}")

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | IoU: {mean_iou:.4f}")

            plt.figure(figsize=(10, 5))
            plt.plot(steps_recorded, loss_values)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid()
            plt.savefig("continued_training_loss.jpg")
            plt.close()

print("Continued training complete. Best model checkpoint saved.")
