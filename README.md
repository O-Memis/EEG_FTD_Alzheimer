# EEG_FTD_Alzheimer
Classification of Alzheimer's , Frontotemporal Dementia and healthy controls by EEG signals. The signals are obtained from an OpenNeuro 2023 dataset :brain: :zap:


<br>


04/2026

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

<br> <br> 

---

## 3) Results 

| **Feature Extraction** | **Shape of the Instances** | **Data Split** | **Model** | **Hyperparameters** | **Accuracy** | **F1** | **Precision** | **Recall** | **10-Fold CV** |
|---|---|---|---|---|---|---|---|---|---|
| CWT | Tensor with 19-channel Matrices (Scalogram Images) | 1 min splits, %72 train, %8 val, %20 test | MobileNetV3 (modified) | Dropout= 0.02, LR schedule= cosine decay, Early stopping patience= 60, Epochs= 150, Batch= 8, Learning rate =0.001, Activation Functions=SiLU, Optimizer=AdamW(momentum=0.9), Loss=Cross Entropy | **Test= %85**, Train= %98 | **Test= %85**, Train= %98 | **Test= %85**, Train= %98 | **Test= %85**, Train= %95 | ... |
| CWT | Tensor with 19-channel Matrices (Scalogram Images) | 1 min splits, %72 train, %8 val, %20 test | ResNet18 (modified) | Dropout= 0.02, LR schedule= cosine decay, Early stopping patience= 60, Epochs= 150, Batch= 8, Learning rate =0.001, Activation Functions=SiLU, Optimizer=AdamW(momentum=0.9), Loss=Cross Entropy | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
| CWT | Tensor with 19-channel Matrices (Scalogram Images) | 1 min splits, %72 train, %8 val, %20 test | DenseNet (modified) | Dropout= 0.02, LR schedule= cosine decay, Early stopping patience= 60, Epochs= 150, Batch= 8, Learning rate =0.001, Activation Functions=SiLU, Optimizer=AdamW(momentum=0.9), Loss=Cross Entropy | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | **Test= %..**, Train= %.. | ... |
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


