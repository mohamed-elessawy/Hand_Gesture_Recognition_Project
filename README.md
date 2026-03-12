# Real-Time Hand Gesture Classification System

This repository contains a complete machine learning pipeline for real-time hand gesture recognition. Built utilizing MediaPipe for spatial feature extraction and Scikit-Learn for classification, the system identifies 18 distinct hand gestures through a live webcam feed. 

## Project Architecture & Organization
The core of this project is built inside `Hand_Landmarks.ipynb`. The notebook is highly organized, fully documented with markdown comments, and contains deep analytical insights. It is separated into 6 logical phases for easy reading and independent execution:

* **Phase 1: Data Loading and Exploration** - Ingesting the 25,000+ sample dataset and analyzing the class distributions.
* **Phase 2: Data Preprocessing** - Applying mathematical translation (wrist-centering) and scale-invariant normalization (scaling by hand size) to the 63 3D coordinates.
* **Phase 3: Model Preparation** - Splitting the data into Train, Validation, and unseen Test sets to prevent data leakage.
* **Phase 4: Model Training** - Each model (SVM, Random Forest, KNN, Logistic Regression) is trained in its own isolated cell for easy testing.
* **Phase 5: Comparing and Evaluating Models** - Evaluating the algorithms against the validation set to find the optimal mathematical fit.
* **Phase 6: Final Evaluation & Live Video Inference** - Testing the winning model on the locked-out test data and deploying the live webcam script.

## Model Evaluation & Selection
Support Vector Machine (SVM) was selected as the deployed production model due to its superior generalization on the hold-out test set. 

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (Deployed)** | **0.9781** | **0.9787** | **0.9781** | **0.9782** |
| Random Forest | 0.9771 | 0.9774 | 0.9771 | 0.9772 |
| KNN | 0.9771 | 0.9774 | 0.9771 | 0.9772 |
| Logistic Regression | 0.8875 | 0.8881 | 0.8875 | 0.8873 |

## How to Run This Project
The lightweight SVM deployment model (`deploy_svm.pkl`) is included directly in this repository for immediate real-time inference.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Live Inference:** Execute the standalone production script in your terminal to launch the webcam feed:
   ```bash
   python live_inference.py
   ```

3. **(Optional) Retrain the Model:** If you wish to explore the data pipeline or retrain the algorithms, open `Hand_Landmarks.ipynb` and execute the pipeline phases to generate a fresh model.

## Live Demo Video
[Watch the real-time inference submission video here](https://drive.google.com/file/d/1YsMbZnnWIRYgRmTTlEsJ8tisZsCCEbAw/view?usp=sharing)