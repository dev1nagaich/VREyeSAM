import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_pixel_metrics(gt_bin, pred_bin):
    TP = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
    FP = np.logical_and(gt_bin == 0, pred_bin == 1).sum()
    FN = np.logical_and(gt_bin == 1, pred_bin == 0).sum()
    TN = np.logical_and(gt_bin == 0, pred_bin == 0).sum()
    return TP, FP, FN, TN

def compute_pixel_level_metrics(gt_folder, pred_folder, thresholds=np.linspace(0.5, 1.0, 11), resize_size=(400, 400)):
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    iou_list = []

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

        TP, FP, FN, TN = calculate_pixel_metrics(gt_bin, pred_bin)

        total_tp += TP
        total_fp += FP
        total_fn += FN
        total_tn += TN

        # Calculate IoU per image
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_list.append(iou)

    # Final pixel-level metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(iou_list)

    return precision, recall, f1, mean_iou

def save_metrics_to_txt(precision, recall, f1, mean_iou, output_txt='pixel_level_segmentation_metrics_CORRECT.txt'):
    with open(output_txt, 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
    print(f"Saved metrics to {output_txt}")

def plot_pixel_metrics(precision, recall, f1, output_plot='pixel_level_metrics_plot_CORRECT.png'):
    labels = ['Precision', 'Recall', 'F1-Score']
    scores = [precision, recall, f1]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, scores, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.05)
    plt.title("Pixel-level Segmentation Metrics")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Annotate the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    print(f"Saved plot to {output_plot}")

if __name__ == "__main__":
    gt_folder = "/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG"
    pred_folder = "/extra/geetanjali/VRBiom/SAM2_FINAL_OUTPUTS/SAM2_original_with_uncertainity_predicted_mask"

    precision, recall, f1, mean_iou = compute_pixel_level_metrics(gt_folder, pred_folder)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    save_metrics_to_txt(precision, recall, f1, mean_iou)
    plot_pixel_metrics(precision, recall, f1)
