# VREyeSAM

**Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss**

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://example.com"><img src="https://img.shields.io/badge/Paper-View-blue.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Demo-Coming%20Soon-orange.svg" alt="Demo"></a>
</p>

---

## 🧠 Description

**VREyeSAM** proposes a robust iris segmentation pipeline for challenging **non-frontal iris images** in virtual reality (VR) environments.  
It fine-tunes the **Segment Anything Model (SAM)** using a novel **uncertainty-weighted loss**, resulting in enhanced segmentation accuracy and generalization.

---

## 👩‍🔬 Authors

- **Geetanjali Sharma**<sup>1</sup>, **Dev Nagachi**<sup>1</sup>, **Gaurav Jaswal**<sup>2</sup>, **Aditya Nigam**<sup>1</sup>, **Raghavendra Ramachandra**<sup>3</sup>  
  <sup>1</sup>IIT Mandi, India |  
  <sup>2</sup>TIH, IIT Mandi, India |  
  <sup>3</sup>NTNU Gjøvik, Norway

---

## 🚀 Highlights

- ✅ Fine-tuned **SAM** for VR iris segmentation.
- ✅ Introduced **uncertainty-weighted loss** to handle prediction ambiguity.
- ✅ Evaluated on **VRBiom** and **CASIA-VR** iris datasets.
- ✅ Achieved **state-of-the-art** segmentation performance.

---

## 📸 Sample Results

<p align="center">
  <img src="assets/sample_output1.png" width="400" />
  <img src="assets/sample_output2.png" width="400" />
</p>

---

## ⚙️ Getting Started

### Installation

```bash
git clone https://github.com/yourusername/VREyeSAM.git
cd VREyeSAM
pip install -r requirements.txt

