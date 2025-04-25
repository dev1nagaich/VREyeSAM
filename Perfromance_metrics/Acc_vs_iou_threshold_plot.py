import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_accuracy_curve(gt_folder, pred_folder, thresholds=np.linspace(0.5, 1.0, 10), resize_size=(400, 400)):
    iou_scores = []

    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

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
        iou_scores.append(iou)

    total = len(iou_scores)
    accuracy = [(np.sum(np.array(iou_scores) >= t) / total) * 100 for t in thresholds]
    return thresholds, accuracy

def plot_accuracy_curves(dataset_results, output_path='accuracy_vs_iou.png'):
    plt.figure(figsize=(8, 6))
    for label, (thresholds, accuracy) in dataset_results.items():
        plt.plot(thresholds, accuracy, label=label, linewidth=2)

    plt.xlabel("Overlapped Threshold IOU")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Overlapped IOU")
    plt.ylim(0, 105)
    plt.xlim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

# Compute for 3 datasets or models
dataset_results = {}

thresholds, acc1 = compute_accuracy_curve(
    gt_folder='/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG',
    pred_folder='/extra/geetanjali/VRBiom/SAM2_FINAL_OUTPUTS/SAM2_original_with_uncertainity_predicted_mask'
)
dataset_results['VREyeSAM'] = (thresholds, acc1)
'''
_, acc2 = compute_accuracy_curve(
    gt_folder='/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG',
    pred_folder='/path/to/CASIA/predicted'
)
dataset_results['CASIA V3 Interval'] = (thresholds, acc2)

_, acc3 = compute_accuracy_curve(
    gt_folder='/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG',
    pred_folder='/path/to/UBIRIS/predicted'
)
dataset_results['UBIRIS-v2'] = (thresholds, acc3)
'''
# Plot
plot_accuracy_curves(dataset_results, output_path='VREyeSAM_accuracy_vs_iou_plot.png')

