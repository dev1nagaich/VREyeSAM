# VREyeSAM : Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss
Geetanjali Sharma<sup>1</sup>, Dev Nagaich<sup>1</sup>, Gaurav Jaswal<sup>2</sup>, Aditya Nigam<sup>1</sup>, Raghavendra Ramachandra<sup>3</sup>  
<sup>1</sup> Indian Institute of Technology Mandi, India
<sup>2</sup>Division of Digital Forensics, Directorate of Forensic Services, Shimla, Himachal Pradesh 171002, India
<sup>3</sup>Norwegian University of Science and Technology (NTNU), Gjøvik, Norway

<p align="center">
  <a href="https://www.researchgate.net/publication/400248367_VREyeSAM_Virtual_Reality_Non-Frontal_Iris_Segmentation_using_Foundational_Model_with_uncertainty_weighted_loss"><img src="https://img.shields.io/badge/Paper-View-blue.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Code-Coming%20Soon-orange" alt="Code"></a>
  <a href="#"><img src="https://img.shields.io/badge/Demo-Coming%20Soon-orange.svg" alt="Demo"></a>
  <a href="#"><img src="https://img.shields.io/badge/VRBiomSegM GT-Coming%20Soon-green.svg" alt="VRBiom-SegM GT"></a>
</p>

**Abstract:** Advancements in virtual and head-mounted devices
have introduced new challenges for iris biometrics, such
as varying gaze directions, partial occlusions, and incon-
sistent lighting conditions. To address these obstacles, we
present VREyeSAM, a robust iris segmentation framework
specifically designed for images captured under both
steady and dynamic gaze scenarios. Our pipeline includes
a quality-aware pre-processing module that automatically
filters out partially or fully closed eyes, ensuring that only
high-quality, fully open iris images are used for training
and inference. In addition, we introduce an uncertainty-
weighted hybrid loss function that adaptively balances
multiple learning objectives, enhancing the robustness of
the model under diverse visual conditions. Evaluated on
the VRBiom dataset, VREyeSAM delivers state-of-the-art
performance, achieving a Precision of 0.751, Recall of
0.870, F1-Score of 0.806, and a mean IoU of 0.647 out-
performing existing segmentation methods by a significant
margin
<p align="center">
  <img src="assets/Teaser_IJCB_UPDATED.png" alt="Architecture" width="600"/>
</p>
Figure: VREyeSAM segmentation on VRBiom non-frontal iris images captured under varied gaze and eyewear conditions. Rows show input images, ground truth masks, and predicted masks with uncertainty maps.

## VREyeSAM Architecture 
<p align="center">
  <img src="assets/VRBIOM_SAM2_UPDATED.png" alt="Architecture" width="600"/>
</p>
Figure: Virtual reality iris biometrics segmentation using transformer based model where network takes binary mask and points as an
prompt as input to train the model with hybrid loss(Focal + Dice + BCE + Uncertainty weighted loss) function. For inference model predict
a binary mask of the input image without explicitly using points as input.

## Results
<p align="center">
  <img src="assets/VREyeSAM_updated_overlay-compressed.png" alt="Architecture" width="600"/>
</p>
Figure: Visual comparison of iris segmentation results across multiple models. The first row displays the original iris images, the second row shows the corresponding ground truth masks, and rows three to seven illustrate the predicted masks from five different segmentation models. Last Overlays column shows GT mask in green, predicted mask in red, and overlapping area i.e both masks agree in yellow color.

## License to use the VRBiom-SegM (Groundtruth Binary Mask)
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/VRBiomSegM GT License-Coming%20Soon-green.svg" alt="VRBiom-SegM GT License"></a>
</p>

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/GeetanjaliGTZ/VREyeSAM
cd VREyeSAM
```
2. Create virtual environment:
```bash
python3 -m venv vreyesam_env
source vreyesam_env/bin/activate  # Linux/Mac
# or source vreyesam_env/Scripts/activate  # Windows git bash
```
3. Install dependencies:
```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/segment-anything-2
cd segment-anything-2
pip install -e .
cd checkpoints
bash download_ckpts.sh
cd ..
```

4. Download VREyeSAM fine-tuned weights:

**Option 1: Using Hugging Face CLI (Recommended)**
```bash
# Download the fine-tuned model weights from Hugging Face
cd segment-anything-2/checkpoints
huggingface-cli download devnagaich/VREyeSAM VREyeSAM_uncertainity_best.torch --local-dir .
cd ../..
```

**Option 2: Direct Download**
- Download directly from Hugging Face: [https://huggingface.co/devnagaich/VREyeSAM](https://huggingface.co/devnagaich/VREyeSAM)
- Download the file: `VREyeSAM_uncertainity_best.torch`
- Place it in: `segment-anything-2/checkpoints/`

**Option 3: Manual Installation**
If you don't have `huggingface-cli`, install it first:
```bash
pip install huggingface-hub
```
Then run the download command from Option 1.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{sharma2025vreyesam,
  title={VREyeSAM: Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss},
  author={Sharma Geetanjali, Nagaich Dev, Jaswal Gaurav, Nigam Aditya, and Ramachandra Raghavendra},
  Conference={IJCB},
  year={2025}
}
