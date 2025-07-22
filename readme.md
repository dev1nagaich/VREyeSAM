# VREyeSAM

**Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss**
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

---

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://example.com"><img src="https://img.shields.io/badge/Paper-View-blue.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Demo-Coming%20Soon-orange.svg" alt="Demo"></a>
</p>

---

## Description

**VREyeSAM** proposes a robust iris segmentation pipeline designed for challenging **non-frontal iris images** in virtual reality (VR) environments.  
It fine-tunes the **Segment Anything Model (SAM)** with a novel **uncertainty-weighted loss**, delivering enhanced segmentation accuracy and generalization across datasets.

---

## Authors

- Geetanjali Sharma<sup>1</sup>, Dev Nagachi<sup>1</sup>, Gaurav Jaswal<sup>2</sup>, Aditya Nigam<sup>1</sup>, Raghavendra Ramachandra<sup>3</sup>  
<sup>1</sup>Indian Institute of Technology Mandi, India  
<sup>2</sup>Technology Innovation Hub, IIT Mandi, India  
<sup>3</sup>NTNU Gjøvik, Norway

---

## Highlights

- ✅ Fine-tuned **SAM** for VR iris segmentation.
- ✅ Introduced **uncertainty-weighted loss** to address prediction ambiguity.
- ✅ Tested on real-world **VRBiom** and **CASIA-VR** iris datasets.
- ✅ Achieved state-of-the-art segmentation performance.

---

## Sample Results

<p align="center">
  <img src="assets/sample_output1.png" width="400" />
  <img src="assets/sample_output2.png" width="400" />
</p>

---

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/VREyeSAM.git
cd VREyeSAM
pip install -r requirements.txt
