# Automatic Detection of Dysarthria Using CNN-GRU on Acoustic Speech Data

| Nama | NRP |
|---------|---------|
| Nabila Shafa Rahayu | 5025241150  |
| Dilbina Windi Azahra | 5025241180  |

https://huggingface.co/spaces/chezeyy/DysarthriaDetec
<img width="1919" height="907" alt="image" src="https://github.com/user-attachments/assets/8049b006-9de4-4b78-9487-cd48ae88ea48" />


# Deskripsi Singkat
Deteksi dysarthria yang menganalisis rekaman suara untuk mengklasifikasikan apakah terdapat gangguan bicara dysarthria atau tidak dengan CNN-GRU yang dilatih menggunakan TORGO Database dengan fitur MFCC sebagai representasi akustik.

# Langkah-Langkah

**Import & Setup**
1. Import library
2. Load dataset dari Kaggle

**EDA**
3. Eksplorasi distribusi label dan gender
4. Cross-tabulation gender × label
5. Preview sample per grup

**Visualisasi Fitur Audio**
6. Waveplot
7. Spectrogram
8. Zero Crossing Rate
9. Spectral Centroids
10. Spectral Rolloff
11. MFCC
12. Mel Spectrogram

**Feature Extraction**
13. Ekstraksi 128 MFCC, dirata-rata per frame
14. Speaker-independent split (train/val/test)
15. Encode label & reshape ke (16×8×1)

**Modelling**
16. Bangun arsitektur CNN-GRU
17. Compile model
18. Training dengan EarlyStopping & ModelCheckpoint

**Evaluasi**
19. Learning curves (loss & accuracy)
20. Classification report
21. ROC curve & AUC score
22. Confusion matrix
23. Recall score

**Deployment**
24. Build Gradio app
25. Deploy ke Hugging Face Space

# Link Artikel Penjelasan 
https://medium.com/@azahrarara06/automatic-detection-of-dysarthria-using-cnn-gru-on-acoustic-speech-data-6ee2be3c4307

# Dataset 
https://www.kaggle.com/datasets/iamhungundji/dysarthria-detection

# Referensi
Shih, D.-H., Liao, C.-H., Wu, T.-W., Xu, X.-Y., & Shih, M.-H. (2022). Dysarthria Speech Detection Using Convolutional Neural Networks with Gated Recurrent Unit.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9602047/

Xu, L., Liss, J., & Berisha, V. (2023). Dysarthria detection based on a deep learning model with a clinically-interpretable layer.
https://pmc.ncbi.nlm.nih.gov/articles/PMC9835557/
