import gradio as gr
import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

def predict(audio_path):
    x, sr = librosa.load(audio_path, sr=22050)
       
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128), axis=1)
    X = mfcc.reshape(1, 16, 8, 1)
    
    prob = model.predict(X)[0][0]
    print(f"Raw prob: {prob:.4f}")
    
    label = "Dysarthria" if prob >= 0.5 else "Non-Dysarthria"
    confidence = prob if prob >= 0.5 else 1 - prob
    return f"{label} (confidence: {confidence:.2%})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav)"),
    outputs=gr.Text(label="Hasil Prediksi"),
    title="Dysarthria Detection",
    description="Upload file audio, model akan mendeteksi apakah terdapat dysarthria atau tidak."
)

demo.launch()
