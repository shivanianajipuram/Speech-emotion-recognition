import os
import tempfile
import numpy as np
import librosa
import streamlit as st
import joblib
from keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎧",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🎧 Speech Emotion Recognition")

st.write("Upload a WAV audio file to predict emotion.")

# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_components():

    try:

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        scaler = joblib.load(
            os.path.join(BASE_DIR, "scaler.pkl")
        )

        selector = joblib.load(
            os.path.join(BASE_DIR, "feature_selector.pkl")
        )

        label_encoder = joblib.load(
            os.path.join(BASE_DIR, "label_encoder.pkl")
        )

        model = load_model(
            os.path.join(BASE_DIR, "emotion_model.h5")
        )

        return scaler, selector, label_encoder, model

    except Exception as e:

        st.error(f"Model Loading Error: {e}")

        st.stop()

scaler, selector, label_encoder, model = load_components()

# =========================================================
# FEATURE SUMMARY
# =========================================================

def summarize_feature(feature):

    return np.hstack([

        np.mean(feature.T, axis=0),

        np.std(feature.T, axis=0),

        np.min(feature.T, axis=0),

        np.max(feature.T, axis=0)

    ])

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_path):

    try:

        audio, sr = librosa.load(audio_path, sr=22050)

        if len(audio) == 0:
            return None

        stft = np.abs(librosa.stft(audio))

        # MFCC
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        mfcc_delta = librosa.feature.delta(mfccs)

        mfcc_delta2 = librosa.feature.delta(
            mfccs,
            order=2
        )

        # CHROMA
        chroma = librosa.feature.chroma_stft(
            S=stft,
            sr=sr
        )

        # CONTRAST
        contrast = librosa.feature.spectral_contrast(
            S=stft,
            sr=sr
        )

        # MEL
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        # TONNETZ
        harmonic = librosa.effects.harmonic(audio)

        tonnetz = librosa.feature.tonnetz(
            y=harmonic,
            sr=sr
        )

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)

        # RMS
        rms = librosa.feature.rms(y=audio)

        # COMBINE MFCC
        mfcc_combined = np.vstack([
            mfccs,
            mfcc_delta,
            mfcc_delta2
        ])

        # FINAL FEATURE VECTOR
        final_features = np.hstack([

            summarize_feature(mfcc_combined),

            summarize_feature(chroma),

            summarize_feature(contrast),

            summarize_feature(mel),

            summarize_feature(tonnetz),

            summarize_feature(zcr),

            summarize_feature(rms)

        ])

        return final_features

    except Exception as e:

        st.error(f"Feature Extraction Error: {e}")

        return None

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(audio_path):

    features = extract_features(audio_path)

    if features is None:
        return None, None

    try:

        # SCALE FEATURES
        scaled_features = scaler.transform([features])

        # FEATURE SELECTION
        selected_features = selector.transform(
            scaled_features
        )

        # PREDICTION
        prediction = model.predict(selected_features)

        predicted_index = np.argmax(prediction)

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = prediction[0][predicted_index] * 100

        return predicted_emotion, confidence

    except Exception as e:

        st.error(f"Prediction Error: {e}")

        return None, None

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Choose WAV File",
    type=["wav"]
)

# =========================================================
# PROCESS AUDIO
# =========================================================

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    # TEMP FILE
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(uploaded_file.read())

        temp_audio_path = temp_audio.name

    # PREDICT
    with st.spinner("Analyzing Emotion..."):

        emotion, confidence = predict_emotion(
            temp_audio_path
        )

    # DELETE TEMP FILE
    os.remove(temp_audio_path)

    # =====================================================
    # SHOW RESULT
    # =====================================================

    if emotion is not None:

        st.success(
            f"Predicted Emotion: {emotion.upper()}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

    else:

        st.error(
            "Could not predict emotion from audio."
        )

else:

    st.info("Please upload a WAV audio file.")

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "Built using Streamlit + TensorFlow + Librosa"
)