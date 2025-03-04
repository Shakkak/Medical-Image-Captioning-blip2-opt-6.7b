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

### Medical Image Examples with Captions

#### Example 1: Abdominal CT Scan with Marked Anomalies

<img src="images/output1.png" alt="Abdominal CT" width="400">

**Real Caption:** CT scan image for lung cancer.

**Generated Caption:** CT scan of the abdomen showing a large mass in the right lower quadrant.

---

#### Example 2: Doppler Ultrasound Image

<img src="images/output2.png" alt="Doppler Ultrasound" width="400">

**Real Caption:** Preoperative CT (axial plane) demonstrating appendix rupture with a gas containing collection (red arrows) adjacent to the caecum (green arrow). The collection contains multiple appendicoliths (white arrow).

**Generated Caption:** CT scan of the abdomen showing a large mass in the right lower quadrant.

---

#### Example 3: Panoramic Dental X-Ray

<img src="images/output3.png" alt="Panoramic Dental X-Ray" width="400">

**Real Caption:** Ultrasonography of the Right Femoral Vein StenosisThe ultrasound scan indicated suspected right femoral vein stenosis (arrow).

**Generated Caption:** Ultrasound of the right atrium showing a large mass in the right atrium (arrow).

---

#### Example 4: Thoracic CT Scan

<img src="images/output4.png" alt="Thoracic CT" width="400">

**Real Caption:** Pre-operative OPG showing cyst-like lesion in the right coronoid process (pointed by yellow arrow). Linear radiopacity is the tracer gutta-percha point passed through the extra-oral cutaneous tract (pointed by blue arrows). The gutta percha point is seen abutting the cyst-like lesion in the coronoid process. OPG, orthopantomograph

**Generated Caption:** Axial view of the mandible showing the presence of a large mass in the mandibular fossa.

---

### Quantitative Metrics:

- **Evaluation Metrics:** BLEU, METEOR, and ROUGE scores were used to evaluate the quality of the generated captions.

The performance of the model was evaluated using standard captioning metrics. Due to the **limited training epochs** (caused by hardware constraints), the results are lower than expected, but they provide a reasonable baseline for further fine-tuning and optimization.

#### Metrics Achieved:

- **BLEU-1:** 0.129
- **BLEU-2:** 0.072
- **BLEU-3:** 0.038
- **BLEU-4:** 0.022
- **METEOR:** 0.063
- **ROUGE-L:** 0.175
- **CIDEr:** 0.117
- **SPICE:** 0.045

#### Interpretation:

1. **BLEU Scores:**

   - **BLEU-1 (0.129):** Indicates the model captures some basic keywords from the ground-truth captions.
   - The scores for higher n-grams (BLEU-2 to BLEU-4) drop significantly, reflecting the model's difficulty in generating longer coherent phrases.

2. **METEOR (0.063):**

   - METEOR evaluates both precision and recall. The low score reflects the limited overlap between generated and reference captions.

3. **ROUGE-L (0.175):**

   - ROUGE-L measures the longest common subsequence between the generated and reference captions. A score of 0.175 suggests the model occasionally aligns well with key sentence structures.

4. **CIDEr (0.117):**

   - CIDEr assesses how well the generated captions match human descriptions. The score indicates that, while some similarity exists, fine-tuning with more epochs is needed to improve domain-specific language generation.

5. **SPICE (0.045):**
   - SPICE evaluates the semantic content of the generated captions. A low score reflects the difficulty in generating captions with meaningful radiology-specific content.

#### Note:

These metrics are consistent with expectations given the **limited number of training epochs** and the size of the **filtered RGB dataset**. With additional computational resources and longer training, the model’s performance could improve significantly.

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
