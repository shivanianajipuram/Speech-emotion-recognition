import joblib
import librosa
import numpy as np
import random
import os # Import os for path checks
from keras.models import load_model

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def summarize_feature(feature):
    return np.hstack([
        np.mean(feature.T, axis=0),
        np.std(feature.T, axis=0),
        np.min(feature.T, axis=0),
        np.max(feature.T, axis=0)
    ])

def augment_audio(audio, sr):
    augmented_versions = []

    # pitch shift
    audio_pitch = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=random.choice([-2, 2])) # Randomize pitch shift steps
    augmented_versions.append(audio_pitch)

    # time stretch
    audio_stretch = librosa.effects.time_stretch(y=audio, rate=random.choice([0.8, 1.2])) # Randomize time stretch rate
    augmented_versions.append(audio_stretch)

    # adding noise
    noise_factor = random.uniform(0.003, 0.008) # Randomize noise factor
    noise = noise_factor * np.random.randn(len(audio))
    audio_noise = audio + noise
    augmented_versions.append(audio_noise)

    return augmented_versions


def features_extract(path_of_files, augment_for_training=False):
  try:
    audio, sr = librosa.load(path_of_files, sr=22050, res_type='scipy')

    #applying one random augmentation feature extraction
    if augment_for_training:
        #get all augmented versions
        augmented_audios = augment_audio(audio, sr)
        #randomly choose one augmented version to extract features from
        audio = random.choice(augmented_audios)

    stft = np.abs(librosa.stft(audio))

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

    if mfccs.shape[1] == 0 or chroma.shape[1] == 0 or contrast.shape[1] == 0 or mel.shape[1] == 0:
        return None  # skip bad files

    mfccs_features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])
    mfccs_features = summarize_feature(mfccs_features)
    chroma = summarize_feature(chroma)
    contrast = summarize_feature(contrast)
    mel = summarize_feature(mel)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    tonnetz = summarize_feature(tonnetz)

    zcr = summarize_feature(librosa.feature.zero_crossing_rate(y=audio))
    rms = summarize_feature(librosa.feature.rms(y=audio))

    return np.hstack([mfccs_features, chroma, contrast, mel, tonnetz, zcr, rms])
  except Exception as e:
    print(f"Error in processing of file {path_of_files} : {e}")

scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
selector = joblib.load("feature_selector.pkl")
model = load_model("emotion_model.h5")

path = input("Enter Path to audio File in .wav format")
new_features = features_extract(path)  # shape: (N,)
new_features_scaled = scaler.transform([new_features])  # shape: (1, full_features)
new_features_selected = selector.transform(new_features_scaled)  # shape: (1, 300)

prediction = model.predict(new_features_selected)
predicted_label = le.inverse_transform([np.argmax(prediction)])
print("Predicted emotion:", predicted_label[0])