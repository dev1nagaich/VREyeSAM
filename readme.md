# VREyeSAM  Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss
- Geetanjali Sharma<sup>1</sup>, Dev Nagachi<sup>1</sup>, Gaurav Jaswal<sup>2</sup>, Aditya Nigam<sup>1</sup>, Raghavendra Ramachandra<sup>3</sup>  
<sup>1</sup> Indian Institute of Technology Mandi, India
<sup>2</sup>Technology Innovation Hub, Indian Institute of Technology Mandi, India
<sup>3</sup>Norwegian University of Science and Technology (NTNU), Gjøvik, Norway

<p align="center">
  <a href="https://example.com"><img src="https://img.shields.io/badge/Paper-View-blue.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Code-Coming%20Soon-orange" alt="Code"></a>
  <a href="#"><img src="https://img.shields.io/badge/Demo-Coming%20Soon-orange.svg" alt="Demo"></a>
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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{sharma2025vreyesam,
  title={VREyeSAM: Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss},
  author={Sharma, Geetanjali and Nagachi, Dev and Jaswal, Gaurav and Nigam, Aditya and Ramachandra, Raghavendra},
  Conference={IJCB},
  year={2025}
}
