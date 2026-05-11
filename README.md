# Speech Emotion Recognition System

## Project Description
This is a machine learning-based web application that predicts human emotions from audio speech. It classifies audio into eight emotions:

angry  
calm  
disgust  
fearful  
happy  
neutral  
sad  
surprised  

The model is trained on extracted audio features and optimized for high validation accuracy and strong generalization performance.

---

## Dataset
The app uses audio files from the **RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)**.  
This dataset contains `.wav` files of actors speaking in different emotional tones. Only speech data (not singing) is used.

---

## Preprocessing Pipeline

### Audio Loading
- `.wav` files loaded using `librosa`
- Sampling rate: 22050 Hz

### Feature Extraction
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma Features
- Spectral Contrast
- Tonnetz

### Data Preparation
- Features combined into a single vector per audio sample
- Stored in pandas DataFrame with labels

### Imbalance Handling
- Dataset was imbalanced across emotion classes  
- Applied upsampling using `sklearn.utils.resample`  
- Balanced all 8 emotion categories  

---

## Feature Engineering

- StandardScaler applied for normalization  
- Label Encoding for emotion classes  
- One-Hot Encoding for training labels  
- Feature selection using SelectKBest (f_classif)  
- Final selected features: **300**

---

## Model Architecture

Input: 300 selected features  

- Dense(512, ReLU, L2 regularization)  
  → BatchNorm + Dropout(0.4)  
- Dense(256, ReLU, L2 regularization)  
  → BatchNorm + Dropout(0.3)  
- Dense(112, ReLU, L2 regularization)  
  → Dropout(0.2)  
- Output Layer: Dense(num_classes, Softmax)

### Training Configuration
- Optimizer: Adam (learning rate = 5e-4)  
- Loss Function: categorical_crossentropy  
- EarlyStopping (patience = 20)  
- ReduceLROnPlateau (factor = 0.2, patience = 10)  
- Class weights applied using `compute_class_weight()`  

---

## Evaluation Metrics

- Test Accuracy: 87.25%  
- Classification Report:
  - Precision
  - Recall
  - F1-score per emotion class  
- Confusion Matrix analysis for prediction performance  

---

## Streamlit Deployment

The application is deployed using Streamlit with a clean UI design:

- Minimal interface (no sidebar clutter)  
- Upload or record `.wav` audio file  
- Displays predicted emotion clearly  
- Shows top-3 emotion probabilities  
- Clean output format (no unnecessary charts or visuals)  

---

## File Structure

```
project/
│
├── app.py                        # Streamlit application
├── features_extract.py          # Audio feature extraction script
├── processed_audio_features.pkl # Preprocessed balanced dataset
│── emotion_model.h5         # Trained deep learning model
│── scaler.pkl               # StandardScaler object
│── selector.pkl             # SelectKBest feature selector (300 features)
│── label_encoder.pkl        # LabelEncoder for emotions
└── requirements.txt             # Dependencies
```
## Download dataset from
```bash
https://zenodo.org/records/1188976
```
---

## Usage Instructions

### 1. Clone Repository
```bash
git clone https://github.com/shivanianajipuram/Speech-emotion-app.git
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```

### 3. Use App
- Upload `.wav` audio file  
- View predicted emotion  
- Check probability distribution  

#Live demo
```bash
https://speech-emotion-recognition-v7t6.onrender.com/
```
##NOTE:
Sample inputs provided in input folder


## Requirements

- Python ≥ 3.8  
- librosa  
- tensorflow  
- scikit-learn  
- numpy  
- pandas  
- matplotlib  
- streamlit  
- joblib  


