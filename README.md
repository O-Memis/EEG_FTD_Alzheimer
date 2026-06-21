# EEG_FTD_Alzheimer
Classification of Alzheimer's , Frontotemporal Dementia and healthy controls by EEG signals. The signals are obtained from an OpenNeuro 2023 dataset :brain: :zap:


<br>


06/2026

<br><br> 


## 1) Data Description

- 19 channel EEG signals sampled at 500Hz
- Total duration of signals varies among subjects

- Resting state & closed eyes recordings are captured by bipolar montage,

- 3 classes, 88 subjects in total
- 36 Alzheimer's, 23 Frontotemporal Dementia, all evaluated by Mini-Mental State Examination (MMSE)
- 29 are healthy controls

- Channel names: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2

- Pre-processed signals are located in the folder "derivatives"

- Their pre-processing steps:

    *1)* BPF 0.5 Hz - 45 Hz
    *2)* Artifact Subspace Reconstruction routine (ASR) [EEG artifact correction method]
    *3)* Independent Component Analysis (ICA) to remove artefacts

- Signals are located in ".set" files, while labels are written in "participants.tsv"

- Dataset link: https://doi.org/10.18112/openneuro.ds004504.v1.0.8

<br> <br> 

---


## 2) Team coding style

### 1.1 Script-first + cell-based style for easier experimentation

We write “notebook-style” `.py` scripts to conduct model experimentations easier:

- Use VS Code / Spyder cell markers: `#%%`
- Run code top-to-bottom in numbered steps
- Avoid forcing a `main()` entrypoint


### 1.2 Minimal abstraction

- Coding style should be different & proper for experimentation stage, and deployment stage.
- Prefer explicit loops and direct code over many helper functions. 
- Create helper functions only when logic is repeated and becomes error-prone. <br><br>

### 1.3 Naming and reproducibility rules

- Common names: `lr`, `batch_size`, `epochs`, `patience`, `device`.
- Keep train/val/test naming consistent. In K-fold CV, call the held-out fold **validation**, not test.
- Fix random seeds for splits.
- Do not redefine the same model class name with different architectures in the same file; use `ModelV1`, `ModelV2` if needed. <br><br>

### 1.4 Minimalist approach

- Minimalism, is better than over-engineering. 
- Do not rewrite/refractor other parts of the code when it is not mandatory. <br><br>

### 1.5 Readable code

- Prefer clear, linear code over clever shortcuts.
- Keep blocks short and avoid deep nesting.
- Keep comments short and concise.
- Document key experiment choices, parameters, functions, and assumptions.
- Keep formatting consistent across the file. <br><br>

---




## 2) Codes 

**eeg_ftd_alzheimer_1.py** : Loads 88-subject EEG recordings, explores the signals by time & frequency plots, segments them into 20-second epochs with stratified subject-wise splits, computes CWT scalograms per channel as feature extraction, and trains a CNN-based classifier. <br> <br>

**eeg_ftd_alzheimer_2.py** : Shares the same data pipeline (loading, segmentation, CWT scalograms). Trains a pretrained YOLO26-cls backbone adapted for 19-channel EEG input (first conv patched 3→19 ch, classifier head patched →3 classes, forward overridden to return raw logits). <br> <br>  

**eeg_ftd_alzheimer_3.py** : One-vs-Rest binary models. Shares the same scalogram pipeline, but treats each class as a binary problem. 3 separate YOLO26m-cls binary models (OvR), same backbone/init as previous, per-model class weights from distribution, then scores by combined probabilities for the final 3-class predictions. <br> <br>  

**eeg_ftd_alzheimer_ammar.py** : Hand-crafted feature extraction pipeline (27 time-domain + 20 frequency-domain + 12 DWT wavelet features per channel, plus 684 inter-channel coherence features = 1 805 features total). Classical ML classifiers (SVM, Random Forest, XGBoost) with PCA (95% variance) and subject-level majority voting. Deep learning models (EEGNet, ShallowConvNet, SpectrogramCNN) on downsampled (128 Hz) raw epochs. All models use subject-wise stratified 70/10/20 train/val/test splits to prevent data leakage. <br> <br>

Simple workflow of the codes: <br> 

```
#%% 1) Imports
#%% 2) Data import and labels
#%% 3) EDA: Basic plots
#%% 4) Frequency analyses for one channel (FFT, Periodogram, Spectrogram, Scalogram)
#%% 5) Dataset preparation: Splitting and segmentation
#%% 6) Data transformation for feature extraction: CWT scalograms
#%% 7) Model training and evaluation
#%% 8) Final test evaluation
#%% Optional 1: Export segment IDs/labels to Excel for manual inspection
#%% Optional 2: Save / load scalograms from disk
```
<br> <br> 

---

## 3) Results 

| **Feature Extraction** | **Shape of the Instances** | **Data Split** | **Model** | **Hyperparameters** | **Accuracy** | **F1** | **Precision** | **Recall** | **10-Fold CV** |
|---|---|---|---|---|---|---|---|---|---|
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | Custom CNN (AlexNet variant) | Dropout= 0.2, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.0001, Weight decay= 0.0005, Activation= SiLU, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.01) | **Test= %63.95** (seg) / **%72.22** (subj), Train= %100.00 | **Test= %62.06** (seg) / **%67.04** (subj), Train= ~%100 | **Test= %61.81** (seg) / **%79.17** (subj), Train= ~%100 | **Test= %63.95** (seg) / **%72.22** (subj), Train= ~%100 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | EfficientNet-B0 (torchvision, modified: first conv 3→19 ch, final linear→3 classes) | Dropout= 0.2, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.0001, Weight decay= 0.0005, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.01) | **Test= %59.25** (seg) / **%55.56** (subj), Train= %100.00 | **Test= %58.19** (seg) / **%55.31** (subj), Train= ~%100 | **Test= %58.28** (seg) / **%55.56** (subj), Train= ~%100 | **Test= %59.25** (seg) / **%55.56** (subj), Train= ~%100 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | Vision Transformer (DeiT-Tiny equivalent: embed_dim=192, heads=6, layers=4, patch=(6×50), 200 patches) | Dropout= 0.4, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.0001, Weight decay= 0.01, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.1) | **Test= %52.90** (seg) / **%61.11** (subj), Train= ~%100.00 | **Test= %50.53** (seg) / **%58.06** (subj), Train= ~%100 | **Test= %52.40** (seg) / **%73.89** (subj), Train= ~%100 | **Test= %52.90** (seg) / **%61.11** (subj), Train= ~%100 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | ResNet50 variant (stem conv 19→64, stages [3,4,6,3] BottleneckEEG, AdaptiveAvgPool, classifier 2048→512→3, SiLU) | Dropout= 0.2, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.0001, Weight decay= 0.001, Activation= SiLU, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.05) | **Test= %58.15** (seg) / **%61.11** (subj), Train= %100.00 | **Test= %56.65** (seg) / **%59.03** (subj), Train= ~%100 | **Test= %55.71** (seg) / **%58.64** (subj), Train= ~%100 | **Test= %58.15** (seg) / **%61.11** (subj), Train= ~%100 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | DenseNet (EEGDenseNet: 4 dense blocks [4,6,8,6 layers], growth_rate=24, GroupNorm, SiLU, dual AdaptiveAvgPool+MaxPool, classifier 672→512→128→3) | Dropout= 0.15, LR schedule= cosine decay (T_max=120), No early stopping, Epochs= 120, Batch= 8, Learning rate= 0.0001, Weight decay= 0.0005, Activation= SiLU, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.05) | **Test= %54.97** (seg) / **%50.00** (subj), Train= %100.00 | **Test= %54.89** (seg) / **%50.00** (subj), Train= ~%100 | **Test= %54.83** (seg) / **%50.00** (subj), Train= ~%100 | **Test= %54.97** (seg) / **%50.00** (subj), Train= ~%100 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | YOLO26s-cls (Ultralytics, modified: first conv 3→19 ch, final linear→3 classes, logits override) | Dropout= 0.10, LR schedule= cosine decay (T_max=120), No early stopping, Epochs= 150, Batch= 8, Learning rate= 0.0001, Weight decay= 0.0001, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.02) | **Test= %67.54** (seg) / **%66.67** (subj), Train= ~%86.41 | **Test= %66.79** (seg) / **%62.38** (subj), Train= ~%85 | **Test= %66.52** (seg) / **%63.89** (subj), Train= ~%85 | **Test= %67.54** (seg) / **%66.67** (subj), Train= ~%85 | ... |
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | YOLO26m-cls (Ultralytics, modified: first conv 3→19 ch via averaged pretrained RGB weights scaled ×3/19, final linear→3 classes, logits override) | Dropout= 0.20, LR schedule= cosine decay (T_max=150), No early stopping, Epochs= 150, Batch= 8, Learning rate= 0.00003, Weight decay= 0.001, Optimizer= AdamW, Loss= Weighted Cross Entropy (label smoothing=0.02), Grad clip= 2.0 | **Test= %63.95** (seg) / **%77.78** (subj), Train= ~%78.05 | **Test= %63.67** (seg) / **%77.08** (subj), Train= ~%78 | **Test= %63.72** (seg) / **%80.25** (subj), Train= ~%78 | **Test= %63.95** (seg) / **%77.78** (subj), Train= ~%78 | ... |
<<<<<<< HEAD
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | **One-vs-Rest (OvR)**: 3× YOLO26m-cls (same backbone & init as above), each trained as binary classifier; per-model class weights from distribution (model_ad: [0.863, 1.189], model_ftd: [0.651, 2.157], model_hc: [0.767, 1.438]); 3-class decision by argmax of combined binary scores. Binary test results — model_ad: Acc=%80.52, F1=%76.38, Prec=%54.70, Rec=%34.78; model_hc: Acc=%78.59, F1=%66.52, Prec=%74.04, Rec=%60.39 | Dropout= 0.20, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.00003, Weight decay= 0.001, Optimizer= AdamW, Loss= Weighted Binary Cross Entropy (label smoothing=0.02), Grad clip= 2.0 | **Test= %72.65** (seg) / **%83.33** (subj), Train= ~%90 | **Test= %71.33** (seg) / **%82.64** (subj), Train= ~%90 | **Test= %72.24** (seg) / **%85.80** (subj), Train= ~%90 | **Test= %72.65** (seg) / **%83.33** (subj), Train= ~%90 | ... |
| Hand-crafted: 27 time-domain + 20 frequency-domain + 12 DWT (db4, level-5) per channel + 171-pair × 4-band inter-channel coherence | Feature vectors: 1 805 features/epoch (1 121 per-channel + 684 coherence), StandardScaler + PCA (95% variance) | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | SVM (RBF kernel, C=10, balanced class weights) | Subject-level majority vote on test set | **Test= %63.3** (seg) / **%66.7** (subj) | **Test= %52.3** (subj, macro) | — | — | ... |
| Hand-crafted: 27 time-domain + 20 frequency-domain + 12 DWT (db4, level-5) per channel + 171-pair × 4-band inter-channel coherence | Feature vectors: 1 805 features/epoch (1 121 per-channel + 684 coherence), StandardScaler + PCA (95% variance) | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | Random Forest (300 trees, balanced class weights) | Subject-level majority vote on test set | **Test= %60.2** (seg) / **%66.7** (subj) | **Test= %51.6** (subj, macro) | — | — | ... |
| Hand-crafted: 27 time-domain + 20 frequency-domain + 12 DWT (db4, level-5) per channel + 171-pair × 4-band inter-channel coherence | Feature vectors: 1 805 features/epoch (1 121 per-channel + 684 coherence), StandardScaler + PCA (95% variance) | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | XGBoost (300 trees, max_depth=6, lr=0.1) | Subject-level majority vote on test set | **Test= %60.6** (seg) / **%72.2** (subj) | **Test= %64.8** (subj, macro) | — | — | ... |
| None — raw EEG signals downsampled 500 Hz → 128 Hz | Tensor (1, 19, 2560) per epoch | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | EEGNet (F1=8, D=2, F2=16, kern=250, Dropout=0.5) | Epochs= 100, Batch= 16, Learning rate= 0.0001, LR schedule= cosine decay (T_max=100), Optimizer= Adam, Loss= Weighted Cross Entropy, Best weights by val loss | **Test= %32.9** (seg) / **%22.2** (subj) | **Test= %18.9** (subj, macro) | — | — | ... |
| None — raw EEG signals downsampled 500 Hz → 128 Hz | Tensor (1, 19, 2560) per epoch | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | ShallowConvNet (40 filters, temporal=25, pool=75/15, square+log activation, Dropout=0.5) | Epochs= 100, Batch= 16, Learning rate= 0.0001, LR schedule= cosine decay (T_max=100), Optimizer= Adam, Loss= Weighted Cross Entropy, Best weights by val loss | **Test= %65.4** (seg) / **%66.7** (subj) | **Test= %52.3** (subj, macro) | — | — | ... |
| Log-power spectrograms per channel (nperseg=256, noverlap=192, 128 Hz) | Tensor (19, 65, 78) per epoch — per-channel spectrogram matrices | 20 sec splits, %70 train, %10 val, %20 test, subject-wise stratified | SpectrogramCNN (3-layer 2D CNN: 32→64→128 filters, AdaptiveAvgPool(4,4), FC 2048→256→3, Dropout=0.5) | Epochs= 100, Batch= 16, Learning rate= 0.0001, LR schedule= cosine decay (T_max=100), Optimizer= Adam, Loss= Weighted Cross Entropy, Best weights by val loss | **Test= %48.9** (seg) / **%50.0** (subj) | **Test= %38.9** (subj, macro) | — | — | ... |
=======
| CWT (cmor-1.0-1.0, 60 scales) | Tensor (19, 60, 1000) — 19-channel scalogram matrices | 20 sec splits, %70 train, %10 val, %20 test | **One-vs-Rest (OvR)**: 3× YOLO26m-cls (same backbone & init as above), each trained as binary classifier; per-model class weights from distribution (model_ad: [0.863, 1.189], model_ftd: [0.651, 2.157], model_hc: [0.767, 1.438]); 3-class decision by argmax of combined binary scores. Binary test results — model_ad: Acc=%80.52, F1=%76.38, Prec=%73.08, Rec=%80.00; model_ftd: Acc=%76.10, F1=%42.52, Prec=%54.70, Rec=%34.78; model_hc: Acc=%78.59, F1=%66.52, Prec=%74.04, Rec=%60.39 | Dropout= 0.20, LR schedule= cosine decay (T_max=200), No early stopping, Epochs= 200, Batch= 8, Learning rate= 0.00003, Weight decay= 0.001, Optimizer= AdamW, Loss= Weighted Binary Cross Entropy (label smoothing=0.02), Grad clip= 2.0 | **Test= %72.65** (seg) / **%83.33** (subj), Train= ~%90 | **Test= %71.33** (seg) / **%82.64** (subj), Train= ~%90 | **Test= %72.24** (seg) / **%85.80** (subj), Train= ~%90 | **Test= %72.65** (seg) / **%83.33** (subj), Train= ~%90 | ... |
>>>>>>> dd849052cda800d8c932938f582e9c23b98010ae
| DWT, Db19, 5 Level Decomposition | Feature Vectors, Features= Normalized Integral, Normalized Band Energy, Spectral Centroid, Median Frequency, Mean Frequency  | 1 min splits, %72 train, %8 val, %20 test | MLP | ... | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
| WPT, Decomposition | Feature Vectors, Features= Normalized Integral, Normalized Band Energy, Spectral Centroid, Median Frequency, Mean Frequency  | 1 min splits, %72 train, %8 val, %20 test | MLP | ... | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
| None | 19-channel Signals | 1 min splits, %72 train, %8 val, %20 test | EEGNet | ... | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
| EMD | ... | 1 min splits, %72 train, %8 val, %20 test | MLP | ... | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
<br> <br> 

---


## 4) Contacs & Referencing 

Please refer to the original source for the dataset. And refer this repository for the codes with the contributors: <br> 

Ammar Omar <br> 
Oğuzhan Memiş - memisoguzhants@gmail.com  


<<<<<<< HEAD

---
=======
>>>>>>> dd849052cda800d8c932938f582e9c23b98010ae
