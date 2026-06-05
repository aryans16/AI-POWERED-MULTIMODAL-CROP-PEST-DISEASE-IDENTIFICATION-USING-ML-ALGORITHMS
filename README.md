# AI-Powered Multimodal Crop Pest & Disease Identification

An AI-based crop disease identification system that combines leaf image analysis with multilingual symptom text input to detect crop diseases more accurately and make agricultural diagnosis more accessible for Indian farmers.

## Overview

Agriculture is a major part of India's economy, but crop diseases and pest infestations cause significant yield losses every year. Traditional crop inspection is slow, inconsistent, and often requires expert knowledge.

Most existing AI-based crop disease systems depend only on images and do not consider the farmer's symptom description, especially in regional languages. This project solves that gap by using a multimodal approach that combines crop leaf images with symptom descriptions in Hindi and English.

## Problem Statement

Existing crop disease detection systems face three major issues:

- Most systems rely only on images, which can cause misclassification when diseases look visually similar.
- Many applications support only English, making them difficult for rural Indian farmers to use.
- Wrong diagnosis can lead to improper pesticide usage, higher cost, and environmental harm.

## Proposed Solution

This project proposes a multimodal AI system that uses:

- **ResNet50 CNN** for image-based crop disease detection.
- **Multilingual BERT (mBERT)** for understanding symptom descriptions in Hindi and English.
- **Late Fusion Network** to combine image features and text embeddings for final disease classification.
- **Recommendation Layer** to map predicted diseases with possible treatment guidance.

## Key Features

- Leaf image-based disease detection.
- Hindi and English symptom text support.
- Multimodal prediction using image + text.
- Supports Tomato, Potato, and Pepper Bell crops.
- Uses transfer learning with ResNet50.
- Uses mBERT for multilingual symptom understanding.
- Lightweight design suitable for normal consumer hardware.
- Designed for future web/mobile deployment.

## Dataset

The project uses the PlantVillage dataset with approximately **8,218 images** across 8 crop disease classes.

### Crop Classes

- Tomato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Potato Healthy
- Potato Early Blight
- Potato Late Blight
- Pepper Bell Healthy
- Pepper Bell Bacterial Spot

The dataset is split into:

- Training set: approximately 6,574 images
- Validation set: approximately 1,644 images

A bilingual symptom CSV dataset is also prepared with:

- Image path
- Disease class
- Symptoms in Hindi
- Symptoms in English

## System Architecture

The system follows a dual-pipeline architecture:

1. User provides a crop leaf image and symptom description.
2. Image is resized to 224x224 and normalized.
3. ResNet50 extracts image features.
4. Symptom text is cleaned and tokenized using mBERT tokenizer.
5. mBERT extracts text embeddings.
6. Image and text features are concatenated.
7. Fully connected layers perform final classification.
8. The system outputs disease class, confidence score, and recommendation.

## Technologies Used

- Python
- PyTorch / TensorFlow
- ResNet50
- Multilingual BERT
- HuggingFace Transformers
- OpenCV
- Pandas
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- VS Code
- GitHub

## Model Details

### Image Model

The image model uses ResNet50 with transfer learning. The final classification layer is modified to classify 8 disease classes.

### Text Model

The text model uses Multilingual BERT to process Hindi and English symptom descriptions. The `[CLS]` token embedding is used for disease classification.

### Multimodal Fusion

The final model combines:

- 2048-dimensional image feature vector from ResNet50
- 768-dimensional text embedding from mBERT

These are merged into a 2816-dimensional feature vector and passed through fully connected layers for classification.

## Results

The model was evaluated on processed test samples.

| Model | Approach | Accuracy |
|---|---|---|
| Image-only Model | ResNet50 Transfer Learning | ~91.49% |
| Text-only Model | Fine-tuned mBERT | Comparable baseline |
| Multimodal Model | ResNet50 + mBERT Late Fusion | ~91.49% |

The multimodal model helps in visually ambiguous cases where image-only prediction may not be enough. Text symptoms provide additional information such as yellowing, wilting, and dark spots.

## Uniqueness

- Combines leaf images with farmer symptom descriptions.
- Supports Hindi and English symptom input.
- Compares image-only, text-only, and multimodal models.
- Focuses on rural usability and low-resource deployment.
- Avoids dependency on complex IoT or chatbot systems.

## Limitations

- Evaluation was performed on a small test set.
- Symptom data was curated from agricultural sources instead of direct farmer input.
- Currently supports only 3 crops and 8 disease classes.
- Real-time edge deployment has not yet been tested.

## Future Enhancements

- Add more crops such as rice, wheat, corn, and soybean.
- Add more Indian languages such as Tamil, Telugu, and Marathi.
- Improve fusion using attention-based models like CLIP or ViLBERT.
- Collect real symptom descriptions from farmers.
- Add treatment recommendation module.
- Build a mobile application with offline inference.
- Optimize model using pruning and quantization for edge devices.

## Team Members

- Aditi Singh - Team Leader
- Aryan Sharma - Co-lead
- Avishi Verma
- K Pooja Sree
- Krrish Ambwani
- Shruti Sharma

## Member Contributions

| Name | Contribution |
|---|---|
| Aditi Singh | Dataset Preparation and Symptom Curation |
| Aryan Sharma | Presentation Design and Project Coordination |
| Avishi Verma | Text Dataset Creation and Language Preprocessing |
| K Pooja Sree | Multimodal Fusion, Training and Evaluation |
| Krrish Ambwani | Text-Based Deep Learning Model |
| Shruti Sharma | Image-Based Deep Learning Model |

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
python app.py
