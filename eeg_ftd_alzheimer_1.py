
"""
Classification of subjects with Alzheimer's disease,
Frontotemporal dementia and Healthy controls by EEG signals.


04/2026

----------------------------------------

Data Description:

    19 channel EEG signals sampled at 500Hz
    Total duration of signals varies among subjects

    Resting state & closed eyes recordings are captured by bipolar montage,

    3 classes, 88 subjects in total
    36 Alzheimer's, 23 Frontotemporal Dementia, all evaluated by Mini-Mental State Examination (MMSE)
    29 are healthy controls

    Channel names: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2

    Pre-processed signals are located in the folder "derivatives"

    Their pre-processing steps:

        1) BPF 0.5 Hz - 45 Hz
        2) Artifact Subspace Reconstruction routine (ASR) [EEG artifact correction method]
        3) Independent Component Analysis (ICA) to remove artefacts

    Signals are located in ".set" files, while labels are written in "participants.tsv"

    Dataset link: https://doi.org/10.18112/openneuro.ds004504.v1.0.8
"""

#%% 1) Imports


from pathlib import Path
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import stft, welch



#%% 2) Data import and labels


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKSPACE_DIR = BASE_DIR.parent if BASE_DIR.name == "EEG_FTD_Alzheimer" else BASE_DIR


DATASET_DIR = WORKSPACE_DIR / "ds004504"
DERIVATIVES_DIR = DATASET_DIR / "derivatives"
PARTICIPANTS_FILE = DATASET_DIR / "participants.tsv"


group_map = {
    "A": "Alzheimer Disease Group",
    "F": "Frontotemporal Dementia Group",
    "C": "Healthy Group",
}


participants_df = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
participants_df = participants_df.rename(columns={"Group": "group_code"})
participants_df["group_name"] = participants_df["group_code"].map(group_map)


records = []

for set_path in sorted(DERIVATIVES_DIR.glob("sub-*/eeg/*.set")):
    participant_id = set_path.parent.parent.name
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")

    signals_df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)
    signals_df.insert(0, "time_sec", raw.times)

    records.append(
        {
            "participant_id": participant_id,
            "file_path": str(set_path),
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
            "n_times": raw.n_times,
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "signals_df": signals_df,
        }
    )



eeg_df = pd.DataFrame(records)
eeg_df = eeg_df.merge(participants_df, on="participant_id", how="left")

eeg_summary = eeg_df[
    [
        "participant_id",
        "group_code",
        "group_name",
        "Gender",
        "Age",
        "MMSE",
        "sfreq",
        "n_channels",
        "n_times",
        "duration_sec",
    ]
].copy()


print(eeg_summary.head())
print()
print(eeg_summary["group_name"].value_counts())


#%% 3) EDA: Basic plots


participant_id = "sub-001"
channel_name = "F4"

subject_row = eeg_df.loc[eeg_df["participant_id"] == participant_id].iloc[0]
subject_signals = subject_row["signals_df"]
time_sec = subject_signals["time_sec"].to_numpy()

channel_columns = [column for column in subject_signals.columns if column != "time_sec"]

print(
    f"{participant_id} | {subject_row['group_name']} | "
    f"{subject_row['duration_sec']:.2f} sec | {subject_row['sfreq']} Hz"
)


#%% 3.1) Plot all channels of one participant

fig, axes = plt.subplots(len(channel_columns), 1, figsize=(14, 2 * len(channel_columns)), sharex=True)

for ax, channel in zip(axes, channel_columns):
    ax.plot(time_sec, subject_signals[channel].to_numpy(), linewidth=0.7)
    ax.set_ylabel(channel)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"All EEG channels | {participant_id} | {subject_row['group_name']}", y=1.0)
plt.tight_layout()
plt.show()


#%% 3.2) Basic plots for one channel

signal = subject_signals[channel_name].to_numpy()
fs = float(subject_row["sfreq"])

plt.figure(figsize=(14, 4))
plt.plot(time_sec, signal, linewidth=0.8)
plt.title(f"Time series | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


fft_values = np.abs(fft(signal))
frequencies = fftfreq(len(signal), d=1 / fs)
positive_mask = frequencies >= 0

frequencies = frequencies[positive_mask]
fft_values = fft_values[positive_mask]
power_db = 10 * np.log10(fft_values**2 + 1e-12)

plt.figure(figsize=(12, 4))
plt.plot(frequencies, fft_values)
plt.title(f"FFT magnitude | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(frequencies, power_db)
plt.title(f"Log power spectrum | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.tight_layout()
plt.show()


nperseg = min(2048, len(signal))
noverlap = nperseg // 2

freq_psd, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")

plt.figure(figsize=(12, 4))
plt.plot(freq_psd, psd)
plt.title(f"Periodogram (Welch PSD) | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.tight_layout()
plt.show()


freq_stft, time_stft, zxx = stft(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(12, 5))
plt.pcolormesh(time_stft, freq_stft, np.abs(zxx), shading="gouraud")
plt.title(f"Spectrogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()


scales = np.geomspace(1, 128, 96)
coefficients, cwt_frequencies = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1 / fs)
log_coefficients = np.log10(np.abs(coefficients) + 1)

sort_index = np.argsort(cwt_frequencies)
cwt_frequencies = cwt_frequencies[sort_index]
log_coefficients = log_coefficients[sort_index]

plt.figure(figsize=(12, 5))
plt.imshow(
    log_coefficients,
    extent=[time_sec[0], time_sec[-1], cwt_frequencies[0], cwt_frequencies[-1]],
    aspect="auto",
    origin="lower",
    cmap="viridis",
)
plt.title(f"Scalogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Log10(Magnitude + 1)")
plt.tight_layout()
plt.show()




