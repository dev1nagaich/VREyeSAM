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

def compute_precision_recall_curve(gt_folder, pred_folder, thresholds=np.linspace(0.5, 1.0, 11), resize_size=(400, 400)):
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    iou_scores = []

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
    precision = []
    recall = []

    for t in thresholds:
        TP = np.sum(np.array(iou_scores) >= t)
        FP = np.sum(np.array(iou_scores) < t)
        precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        recall.append(TP / total if total > 0 else 0)

    return thresholds, precision, recall
    
    
def plot_precision_recall_all(dataset_results, output_path='precision_recall_vs_iou.png'):
    plt.figure(figsize=(10, 6))

    width = 0.08
    x = np.arange(len(next(iter(dataset_results.values()))[0]))  # from thresholds

    for i, (label, (thresholds, precisions, recalls)) in enumerate(dataset_results.items()):
        shift = (i - 1) * width
        plt.bar(x + shift, precisions, width=width, label=f'{label} Precision')
        plt.plot(x + shift, recalls, linestyle='--', linewidth=2, label=f'{label} Recall')

    threshold_labels = [f"{t:.2f}" for t in thresholds]
    plt.xticks(x, threshold_labels, rotation=45)
    plt.ylim(0, 1.05)
    plt.xlabel("Overlapped Threshold IOU")
    plt.ylabel("Precision / Recall")
    plt.title("Precision-Recall vs Overlapped IOU")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

dataset_results = {}

thresholds, prec1, rec1 = compute_precision_recall_curve(
    gt_folder='/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG',
    pred_folder='/extra/geetanjali/VRBiom/SAM2_FINAL_OUTPUTS/SAM2_original_with_uncertainity_predicted_mask'
)

dataset_results['IITD'] = (thresholds, prec1, rec1)


'''
_, prec2, rec2 = compute_precision_recall_curve(
    gt_folder='/path/to/CASIA/gt',
    pred_folder='/path/to/CASIA/pred'
)
dataset_results['CASIA V3 Interval'] = (thresholds, prec2, rec2)

_, prec3, rec3 = compute_precision_recall_curve(
    gt_folder='/path/to/UBIRIS/gt',
    pred_folder='/path/to/UBIRIS/pred'
)
dataset_results['UBIRIS-v2'] = (thresholds, prec3, rec3)
'''
plot_precision_recall_all(dataset_results, output_path='VREyeSAM_precision_recall_vs_iou_plot.png')

