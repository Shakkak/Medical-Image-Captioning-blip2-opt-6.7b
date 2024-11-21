# Medical Image Captioning with BLIP2-OPT-6.7B

## Overview

This repository focuses on the application of **BLIP2-OPT-6.7B** for generating descriptive captions for radiology images. The project leverages a vision-language model fine-tuned on a medical dataset, **Radiology Object in Context Version 2 (ROCOv2)**, for medical image captioning. To overcome computational constraints, **Parameter-Efficient Fine-Tuning (PEFT)** was utilized, enabling efficient adaptation of the large model to the medical domain.

---

## Disclaimer

This project was conducted **solely for research and learning purposes**. Due to **limited computational resources**, the model was fine-tuned with a reduced number of epochs using a **single NVIDIA V100 GPU**. To mitigate resource limitations, **PEFT (Parameter-Efficient Fine-Tuning)** was employed to efficiently train the model on the medical dataset. The results reflect the simplified approach and are not intended for real-world clinical applications.

---

## Dataset: ROCOv2

The **Radiology Object in Context Version 2 (ROCOv2)** dataset provides a large-scale collection of radiology images annotated with descriptive captions. It is divided into three subsets: **train**, **validate**, and **test**.

### Dataset Size

- **Original Dataset:**

  - **Training Set:** 59,958 images
  - **Validation Set:** 9,904 images
  - **Test Set:** 9,927 images

- **Filtered Dataset (RGB-Only):**
  - **Training Set:** 37,330 images
  - **Validation Set:** 6,672 images
  - **Test Set:** 6,780 images

To simplify processing and reduce computational complexity, only RGB images were selected from the dataset, resulting in a reduced dataset size.

---

## Model: BLIP2-OPT-6.7B

BLIP2 (Bootstrapped Language-Image Pre-training) is a cutting-edge vision-language model that combines image and text representations. For this project, the **BLIP2-OPT-6.7B** variant was used, featuring:

- **Vision Encoder:** Pretrained visual backbone for feature extraction.
- **Language Model:** OPT-6.7B, optimized for generating textual captions.
- **Parameter-Efficient Fine-Tuning (PEFT):** Applied to adapt the large model for the medical domain without fully training all parameters, significantly reducing computational costs.

---

## Methodology

### Preprocessing

1. **Dataset Filtering:**

   - Only RGB images from the ROCOv2 dataset were included to simplify training.
   - Images were resized and normalized during preprocessing.

2. **Caption Tokenization:**
   - Captions were tokenized into a format compatible with the BLIP2-OPT model.

### Training

1. **Fine-Tuning with PEFT:**

   - **Parameter-Efficient Fine-Tuning (PEFT):**
     - Only a subset of model parameters was fine-tuned while the majority of the pre-trained parameters were frozen.
     - Techniques like **LoRA (Low-Rank Adaptation)** were employed to adapt the model to the medical domain efficiently.
   - **Hardware:** Training was conducted on a single **NVIDIA V100 GPU**.
   - **Training Epochs:** Reduced number of epochs to accommodate hardware constraints.
   - **Optimizer and Scheduler:** Configured for medical domain fine-tuning.
   - **Batch Size:** Adjusted based on GPU memory limits.

2. **Validation:**

   - Performance was monitored on the validation set during training to prevent overfitting.

3. **Testing:**
   - The model's final performance was assessed on the test set, with a focus on generating accurate captions.

---

## Results

### Quantitative Metrics:

- **Evaluation Metrics:** BLEU, METEOR, and ROUGE scores were used to evaluate the quality of the generated captions.
- **Performance:** Despite limited training epochs and computational constraints, the model demonstrated reasonable performance on both validation and test sets.

### Qualitative Observations:

- The model successfully generated meaningful captions for many radiology images, especially for simpler cases.
- **Challenges:**
  - Complex radiological findings occasionally led to incomplete or less accurate captions.
  - Anatomical overlaps in some images impacted caption quality.

---

## Benefits of Using PEFT

- **Efficient Training:** By fine-tuning only a fraction of the parameters, PEFT significantly reduced the computational and memory overhead.
- **Adaptability:** The pre-trained model was adapted to the medical domain without requiring full-scale retraining.
- **Resource Optimization:** Enabled the use of a large-scale model like BLIP2-OPT-6.7B on a single NVIDIA V100 GPU.

---

## Limitations

1. **Limited Training Epochs:**
   - Due to computational constraints, the model was trained with a smaller number of epochs, which may have limited its performance.
2. **Dataset Filtering:**
   - Only RGB images were used, reducing the diversity of the dataset and potentially impacting the model's generalization.
3. **Hardware Constraints:**
   - Training was conducted on a single **NVIDIA V100 GPU**, restricting the scale of experiments.

---

## Future Work

- **Expand Dataset Coverage:** Include grayscale images to improve dataset diversity.
- **Optimize Training:** Use additional GPUs or cloud-based resources to allow longer training sessions and more hyperparameter tuning.
- **Explore Larger Architectures:** Experiment with larger vision-language models to improve caption generation quality.
- **Advanced PEFT Techniques:** Experiment with more advanced parameter-efficient techniques to further optimize training performance.

---

## Acknowledgments

- **Model:** [BLIP2-OPT-6.7B](https://huggingface.co/Salesforce/blip2-opt-6.7b)
- **Dataset:** [ROCOv2](https://github.com/radiclinic/roco)

This project serves as a demonstration of medical image captioning using state-of-the-art vision-language models and is not intended for real-world clinical applications.
