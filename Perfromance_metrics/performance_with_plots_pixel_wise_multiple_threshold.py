import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_iou(gt_bin, pred_bin):
    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    return intersection / union if union != 0 else 1.0 if intersection == 0 else 0.0

def compute_pixel_metrics_per_threshold(gt_folder, pred_folder, thresholds=np.linspace(0.5, 1.0, 11), resize_size=(400, 400)):
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    # Store all image-level IoUs
    image_ious = []
    gt_bin_list = []
    pred_bin_list = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt is None or pred is None:
            continue

        gt = cv2.resize(gt, resize_size)
        pred = cv2.resize(pred, resize_size)

        _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
        _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

        iou = calculate_iou(gt_bin, pred_bin)

        image_ious.append(iou)
        gt_bin_list.append(gt_bin.flatten())
        pred_bin_list.append(pred_bin.flatten())

    image_ious = np.array(image_ious)
    gt_bin_list = np.array(gt_bin_list)
    pred_bin_list = np.array(pred_bin_list)

    precision_list = []
    recall_list = []
    f1_list = []

    for t in thresholds:
        valid_idx = np.where(image_ious >= t)[0]

        if len(valid_idx) == 0:
            precision = recall = f1 = 0.0
        else:
            selected_gt = gt_bin_list[valid_idx].flatten()
            selected_pred = pred_bin_list[valid_idx].flatten()

            TP = np.sum((selected_gt == 1) & (selected_pred == 1))
            FP = np.sum((selected_gt == 0) & (selected_pred == 1))
            FN = np.sum((selected_gt == 1) & (selected_pred == 0))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return thresholds, precision_list, recall_list, f1_list

def save_metrics_to_csv(thresholds, precision, recall, f1, output_csv='VREyeSAM_pixel_level_threshold_metrics_28_april.csv'):
    df = pd.DataFrame({
        'IoU Threshold': thresholds,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to: {output_csv}")

def plot_metrics(thresholds, precision, recall, f1, output_plot='VREyeSAM_pixel_level_threshold_metrics_plot_28_april.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, marker='o', label="Precision", linewidth=2)
    plt.plot(thresholds, recall, marker='^', label="Recall", linewidth=2)
    plt.plot(thresholds, f1, marker='s', label="F1-Score", linewidth=2)

    plt.xlabel('IoU Threshold')
    plt.ylabel('Score')
    plt.title('Pixel-level Metrics vs IoU Threshold')
    plt.ylim(0, 1.05)
    plt.xlim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    print(f"Saved plot to: {output_plot}")

if __name__ == "__main__":
    gt_folder = "/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG"
    pred_folder = "/extra/geetanjali/VRBiom/SAM2_FINAL_OUTPUTS/SAM2_original_with_uncertainity_predicted_mask"

    thresholds, precision_list, recall_list, f1_list = compute_pixel_metrics_per_threshold(gt_folder, pred_folder)

    save_metrics_to_csv(thresholds, precision_list, recall_list, f1_list, output_csv="VREyeSAM_performacne_updated_pixel_level_threshold_metrics_28_april.csv")

    plot_metrics(thresholds, precision_list, recall_list, f1_list, output_plot="VREyeSAM_performance_updated_pixel_level_threshold_metrics_plot_28_april.png")

