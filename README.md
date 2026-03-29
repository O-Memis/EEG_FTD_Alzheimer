# EEG_FTD_Alzheimer
Classification of Alzheimer's , Frontotemporal Dementia and healthy controls by EEG signals. The signals are obtained from an Openneuro 2023 dataset :brain: :zap:


<br>


04/2026

<br> 


## Data Description

- 19 channel EEG signals sampled at 500Hz
- Total duration of signals varies among subjects

- Resting state & closed eyes recordings are captured by bipolar montage,

- 3 classes, 88 subjects in total
- 36 Alzheimer's, 23 Frontotemporal Dementia, all evaluated by Mini-Mental State Examination (MMSE)
- 29 are healthy controls

- Channel names: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2

- Pre-processed signals are located in the folder "derivatives"

- Their pre-processing steps:

    1) BPF 0.5 Hz - 45 Hz
    2) Artifact Subspace Reconstruction routine (ASR) [EEG artifact correction method]
    3) Independent Component Analysis (ICA) to remove artefacts

- Signals are located in ".set" files, while labels are written in "participants.tsv"

- Dataset link: https://doi.org/10.18112/openneuro.ds004504.v1.0.8