import os
import cv2
import numpy as np

def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union != 0 else 1.0 if intersection == 0 else 0.0

def compute_metrics_from_iou(gt_folder, pred_folder, output_txt='metrics_from_iou_threshold.txt', thresholds=np.linspace(0.5, 1.0, 11), resize_size=(400, 400)):
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

    def safe_div(a, b):
        return a / b if b != 0 else 0

    with open(output_txt, 'w') as f:
        f.write("IOU_Threshold\tPrecision\tRecall\tF1-Score\tType_I\tType_II\n")
        for t in thresholds:
            TP = sum(iou >= t for iou in iou_scores)
            FP = sum(iou < t for iou in iou_scores)
            FN = total - TP
            TN = 0  # Not used here but conceptually FP + TN = total

            precision = safe_div(TP, TP + FP)
            recall = safe_div(TP, TP + FN)
            f1 = safe_div(2 * precision * recall, precision + recall)
            type_I = safe_div(FP, total)
            type_II = safe_div(FN, TP + FN)

            f.write(f"{t:.2f}\t\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t\t{type_I:.4f}\t{type_II:.4f}\n")

    print(f"Metrics saved to {output_txt}")

compute_metrics_from_iou(
    gt_folder='/extra/geetanjali/VRBiom/VR_Biom_test_dataset/test/GT_TEST_ORIGINAL_PNG',
    pred_folder='/extra/geetanjali/VRBiom/SAM2_FINAL_OUTPUTS/SAM2_original_with_uncertainity_predicted_mask',
    output_txt='/extra/geetanjali/VRBiom/Performance_evaluation_for_VRBiom_segmentation_table/VREyeSAM_pixel_metrics_by_iou_threshold.txt',
    thresholds=np.linspace(0.5, 1.0, 11),
    resize_size=(400, 400)
)
