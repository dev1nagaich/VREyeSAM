import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt


# Set input and output directories
images_dir = "/VREyeSAM/VR_Biom_test_dataset/test/images"
output_folder = "/VREyeSAM/VREyeSAM_recognition"
uncertainty_output_folder = os.path.join(output_folder, "test_uncertainty_maps")
os.makedirs(uncertainty_output_folder, exist_ok=True)

# Load model
sam2_checkpoint = "/VREyeSAM/segment-anything-2/sam2/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
FINE_TUNED_MODEL_WEIGHTS = "/VREyeSAM/segment-anything-2/fine_tuned_sam2_with_uncertainity_best.torch"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))


def read_image(image_path):
    img = cv2.imread(image_path)[..., ::-1]  # BGR to RGB
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img


def generate_random_points(image, num_points=30):
    h, w, _ = image.shape
    points = []
    for _ in range(num_points):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        points.append([[x, y]])
    return np.array(points)


def perform_inference(images_dir, output_folder, uncertainty_output_folder):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        image = read_image(image_path)
        input_points = generate_random_points(image, num_points=30)

        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1]),
                return_logits=True  # make sure it returns logits
            )

        # Use best prediction
        best_index = np.argmax(scores[:, 0])
        best_mask = masks[best_index, 0]
        best_logit = logits[best_index, 0]  # shape: (H, W)

        # Compute uncertainty: U = 1 - |2*p - 1| ; where p = sigmoid(logit)
        prob = 1 / (1 + np.exp(-best_logit))  # sigmoid
        uncertainty = 1 - np.abs(2 * prob - 1)  # near 0.5 is high uncertainty
        
        # Normalize the uncertainty map to range 0–255
        uncertainty_map = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        uncertainty_map_uint8 = (uncertainty_map * 255).astype(np.uint8)

        # Apply colormap (e.g., COLORMAP_JET or COLORMAP_PLASMA)
        colored_uncertainty = cv2.applyColorMap(uncertainty_map_uint8, cv2.COLORMAP_PLASMA)
        # Save with OpenCV
        unc_output_path = os.path.join(uncertainty_output_folder, image_name.rsplit('.', 1)[0] + '_uncertainty.png')
        cv2.imwrite(unc_output_path, colored_uncertainty)

        print(f"Saved uncertainty map: {unc_output_path}")


perform_inference(images_dir, output_folder, uncertainty_output_folder)

