
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
from ultralytics import YOLO
from scipy.fft import fft, fftfreq
from scipy.signal import stft, welch



#%% 2) Data import and labels



# 2.1) Functions to identify the base directory address 
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKSPACE_DIR = BASE_DIR.parent if BASE_DIR.name == "EEG_FTD_Alzheimer" else BASE_DIR



# 2.2) Directories
DATASET_DIR = WORKSPACE_DIR / "ds004504"
DERIVATIVES_DIR = DATASET_DIR / "derivatives"
PARTICIPANTS_FILE = DATASET_DIR / "participants.tsv"



# 2.3) Dictionary to match the data with labels
group_map = {
    "A": "Alzheimer Disease Group",
    "F": "Frontotemporal Dementia Group",
    "C": "Healthy Group",
}


"""
Groups were written as A,F,C corresponding to the class names. 
These names will be gathered and made readable in the dataframe.
"""



participants_df = pd.read_csv(PARTICIPANTS_FILE, sep="\t")                   # collecting participant (subject) information
participants_df = participants_df.rename(columns={"Group": "group_code"})    # column name change
participants_df["group_name"] = participants_df["group_code"].map(group_map) # new column to match them with labels




# 2.4) Loop to gather the data

records = []

for set_path in sorted(DERIVATIVES_DIR.glob("sub-*/eeg/*.set")):            # look into this directory, by this structure 
    
    participant_id = set_path.parent.parent.name                            # get sub-0xx info
    
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")   # read .set file by using MNE library. 
                                                                            # EEGLAB object, it includes relevant methods and information.


    signals_df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names)       # gather 19 channel signals (with column names) into a dataframe 
    
    signals_df.insert(0, "time_sec", raw.times)                             # add time information, will be used to divide the signals


    # to collect all types of information into a list
    
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



# 2.5) Generate the new dataframe to gather all the information and data together.


eeg_df = pd.DataFrame(records)                                           # convert this ordered list into dataframe

eeg_df = eeg_df.merge(participants_df, on="participant_id", how="left")  # order this dataframe by ID


del records # free 2.73 GB space from the env



#%% 3) EDA: Basic plots


# Selection of the subject
participant_id = "sub-001"
channel_name = "F4"


# 3.1) Select the entire row of eeg_df for selected subject

subject_row = eeg_df.loc[eeg_df["participant_id"] == participant_id].iloc[0]  
 

# 3.2) Extract the full signal for that subject, column names are the channels
subject_signals = subject_row["signals_df"]         
time_sec = subject_signals["time_sec"].to_numpy()


# 3.3) Get the column names
channels = subject_row["channel_names"]


print(
    f"{participant_id} | {subject_row['group_name']} | "
    f"{subject_row['duration_sec']:.2f} second duration | {subject_row['sfreq']} Hz | "
    f"{subject_row['Age']} age | {subject_row['Gender']} gender | {subject_row['MMSE']} MMSE score "
)



# 3.4) Plot all channels of one participant

fig, axes = plt.subplots(19, 1, figsize=(76,38), sharex=True)                # generate subplots

for ax, channel in zip(axes, channels):                                      # select by channel names
    ax.plot(time_sec, subject_signals[channel].to_numpy(), linewidth=0.6)    # extract signal arrays
    ax.set_ylabel(channel)
    ax.set_ylim(-0.00015, 0.00015)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"All EEG channels | {participant_id} | {subject_row['group_name']}", y=1.0)
plt.tight_layout()
plt.show()


#%% 4.1) Frequency analyses for one channel

signal = subject_signals[channel_name].to_numpy()*10000  # Volts to 0.1mV scale 
fs = 500


#  Time domain

plt.figure(figsize=(57, 10))
plt.plot(time_sec, signal, linewidth=0.8)
plt.title(f"Time series | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()



#  Frequency domain: FFT

fft_values = np.abs(fft(signal))
frequencies = fftfreq(len(signal), d=1 / fs)   # linear x-axis for frequencies
positive_mask = frequencies >= 0               # one-sided

frequencies = frequencies[positive_mask]
fft_values = fft_values[positive_mask]
power_db = 10 * np.log10(fft_values**2 + 1e-12)


# Linear plot with magnitudes
plt.figure(figsize=(20, 10))
plt.plot(frequencies, fft_values)
plt.title(f"FFT magnitude | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-1,60)              # focus on first 60 Hz part, others are not shown
plt.tight_layout()
plt.show()


# Linear plot with (logarithmic) power
plt.figure(figsize=(20, 10))
plt.plot(frequencies, power_db)
plt.title(f"Log power spectrum | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.xlim(-1,60) 
plt.tight_layout()
plt.show()


#%% 4.2) Frequency domain: Periodogram
    
nperseg = 4096
noverlap = nperseg // 4

freq_psd, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")

plt.figure(figsize=(15, 5))
plt.plot(freq_psd, psd)
plt.title(f"Periodogram (Welch PSD) | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(-1,60)  
plt.tight_layout()
plt.show()


#%% 4.3) Time-frequency analysis: Spectrogram

freq_stft, time_stft, zxx = stft(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(20, 15))
plt.pcolormesh(time_stft, freq_stft, np.abs(zxx), shading="gouraud")
plt.title(f"Spectrogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylim(0,30)                     # display up to 30 Hz 
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()


#%% 4.4) Time-frequency analysis: Scalogram

scales = np.linspace(11, 500, 60)  # from 1 Hz to 45 Hz , 60 frequencies

coefficients, cwt_frequencies = pywt.cwt(signal, scales, "cmor-1.0-1.0", sampling_period=1 / fs)
log_coefficients = 10*np.log10(np.abs(coefficients) + 1)

sort_index = np.argsort(cwt_frequencies)
cwt_frequencies = cwt_frequencies[sort_index]
log_coefficients = log_coefficients[sort_index]

plt.figure(figsize=(20, 15))
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



#%% 4.5) 20-seconds segmented plots



signal_segment = signal[0:fs*20]
time_segment = time_sec[0:fs*20]



#  Time domain
plt.figure(figsize=(15, 5))
plt.plot(time_segment, signal_segment, linewidth=2)
plt.title(f"20 seconds segment | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (0.1mV scale)")
plt.tight_layout()
plt.show()



# Frequency domain
nperseg = 512
noverlap = nperseg // 4
freq_psd, psd = welch(signal_segment, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann")

plt.figure(figsize=(15, 5))
plt.plot(freq_psd, psd, linewidth=2)
plt.title(f"Periodogram (Welch PSD) | {participant_id} | {channel_name}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(-1,45)  
plt.tight_layout()
plt.show()




# TF 1: Spectrogram
freq_stft, time_stft, zxx = stft(signal_segment, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(12, 10))
plt.pcolormesh(time_stft, freq_stft, np.abs(zxx), shading="gouraud")
plt.title(f"Spectrogram | {participant_id} | {channel_name}")
plt.xlabel("Time (s)")
plt.ylim(0,45)                     
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()




# TF 2: Scalogram 
coefficients, cwt_frequencies = pywt.cwt(signal_segment, scales, "cmor-1.0-1.0", sampling_period=1 / fs)
log_coefficients = 10*np.log10(np.abs(coefficients) + 1)

sort_index = np.argsort(cwt_frequencies)
cwt_frequencies = cwt_frequencies[sort_index]
log_coefficients = log_coefficients[sort_index]

plt.figure(figsize=(12, 10))
plt.imshow(
    log_coefficients,
    extent=[time_segment[0], time_segment[-1], cwt_frequencies[0], cwt_frequencies[-1]],
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




#%% 5.1) Dataset preparation: Splitting and segmentation


"""
4 criteria for the separation; 

1) Stratified sampling to maintain class balance in each set, 
2) Subject-wise separation to prevent data leakage, 
3) Randomization to ensure that the segments are randomly distributed across the sets.
4) The total number of segments in each set should be approximately 70%, 10%, and 20% of the total segments, respectively. 

Dataset splitting is handled by 2 steps:
    Train and temporary sets
    Temporary set is splitted into test and validation 
"""


# 5.1) Set the segment (epoch) length
segment_sec = 20
segment_points = int(segment_sec * 500)                         # 20 sec * 500 Hz = 10000 points
n_channels = 19                                                 # 19 EEG channels




# 5.2) Split the subjects first, by stratified random assignment

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Extract subject IDs and their class labels
subject_ids = eeg_df["participant_id"].to_numpy()              # 88 subject IDs
subject_labels = eeg_df["group_code"].to_numpy()               # 88 labels (A, F, or C)



# First split: 70% train, 30% temporary (which will become val + test in the next part)
train_subject_ids, temp_subject_ids, _, _ = train_test_split(
    subject_ids,
    subject_labels,
    test_size=0.30,
    stratify=subject_labels,             # stratify respect to labels, for maintaining class balance
    random_state=42,
)




# Second split: divide 30% temp into 10% val and 20% test

temp_labels = eeg_df.set_index("participant_id").loc[temp_subject_ids, "group_code"].to_numpy()

val_subject_ids, test_subject_ids, _, _ = train_test_split(
    temp_subject_ids,
    temp_labels,
    test_size=2 / 3,                                            # 2/3 of 30% = 20% test
    stratify=temp_labels,
    random_state=42,
)




# 5.3) Generate containers for segments in each split

train_data_list = []                # will hold (10000, 19) segment signals
val_data_list = []
test_data_list = []

train_segment_ids = []              # will hold subject IDs per segment
val_segment_ids = []
test_segment_ids = []

train_segment_labels = []           # will hold labels per segment (respect to subject ID)
val_segment_labels = []
test_segment_labels = []



# Convert IDs to sets for membership checking during loop (faster)

train_subject_id_set = set(train_subject_ids)                   # {sub-042, sub-061, ...}
val_subject_id_set = set(val_subject_ids)
test_subject_id_set = set(test_subject_ids)




# 5.4) Iterate all 88 subjects, segment each signal, and route to correct split


for _, subject_row in eeg_df.iterrows():    # look for every row of this dataframe

    
    # Extract current subject's information
    participant_id = subject_row["participant_id"]
    group_code = subject_row["group_code"]
    channel_names = subject_row["channel_names"]
    signals = subject_row["signals_df"][channel_names].to_numpy()   # shape: (total_points, 19)
    n_segments = subject_row["n_times"] // segment_points           # determination of total segment count


    # Determine the split of this subject belongs to (train, test, or val)
    
    if participant_id in train_subject_id_set:
        
        target_data_list = train_data_list
        target_id_list = train_segment_ids
        target_label_list = train_segment_labels
        
    elif participant_id in val_subject_id_set:
        
        target_data_list = val_data_list
        target_id_list = val_segment_ids
        target_label_list = val_segment_labels
        
    else:
        target_data_list = test_data_list
        target_id_list = test_segment_ids
        target_label_list = test_segment_labels


    # Prepare all 60-second segments for this subject
    
    for segment_index in range(n_segments):  # total segment count we determined above  (9, 13 etc.)


        # Extract 60*500 points in order
        start = segment_index * segment_points   # index * 10000  (first index=0)
        end = start + segment_points
        segment = signals[start:end, :]          # extract this segment from subject's signals (all 19 channels)     
        
        
        # shape of each segment: (10000, 19)


        # Append segment and metadata to the appropriate split
        target_data_list.append(segment)
        target_id_list.append(participant_id)
        target_label_list.append(group_code)


# We have obtained all the segments/epochs in a list




# 5.5) Convert lists to arrays (3D tensors)
# (n_segments, 10000, 19)

train_data = np.stack(train_data_list).astype(np.float32) if train_data_list else np.empty((0, segment_points, n_channels), dtype=np.float32)
val_data   = np.stack(val_data_list).astype(np.float32)   if val_data_list   else np.empty((0, segment_points, n_channels), dtype=np.float32)
test_data  = np.stack(test_data_list).astype(np.float32)  if test_data_list  else np.empty((0, segment_points, n_channels), dtype=np.float32)

# ---- reduced storage size by float64 to float32



# segment IDs to arrays 
train_segment_ids = np.array(train_segment_ids)
val_segment_ids = np.array(val_segment_ids)
test_segment_ids = np.array(test_segment_ids)


# label arrays 
train_segment_labels = np.array(train_segment_labels)
val_segment_labels = np.array(val_segment_labels)
test_segment_labels = np.array(test_segment_labels)




# 5.6) Final shuffle on segments in each split (prevent order bias & obtain better mixture)

train_data, train_segment_ids, train_segment_labels = shuffle(
    train_data,
    train_segment_ids,
    train_segment_labels,
    random_state=42,              # reproducibility by RNG
)

val_data, val_segment_ids, val_segment_labels = shuffle(
    val_data,
    val_segment_ids,
    val_segment_labels,
    random_state=42,
)

test_data, test_segment_ids, test_segment_labels = shuffle(
    test_data,
    test_segment_ids,
    test_segment_labels,
    random_state=42,
)



print("train_data shape:", train_data.shape)
print("val_data shape:", val_data.shape)
print("test_data shape:", test_data.shape)





# ----Rescale the signal values

"""
MNE reads EEGLAB .set files in Volts; but the device sensitivity is 10 uV/mm (see dataset README)
so the native physical unit is microvolts. Multiplying by 1e6 brings the
signal into uV range (approx +-150 uV), which is the standard EEG amplitude scale.

Multiply with 1*10^6 for -> uV scale
Multiply with 1*10^3 for -> mV scale
"""
scale_factor = 10000                                             
train_data   = train_data * scale_factor
val_data     = val_data   * scale_factor
test_data    = test_data  * scale_factor





# 5.8) Delete temporary variables to free memory


del (
    subject_ids,
    subject_labels,
    temp_subject_ids,
    temp_labels,
    train_subject_ids,
    val_subject_ids,
    test_subject_ids,
    train_subject_id_set,
    val_subject_id_set,
    test_subject_id_set,
    train_data_list,
    val_data_list,
    test_data_list,
    eeg_df
)



#%% 5.2) Class counts by subjects and by segments


# Subject-level class counts (derived from unique subject IDs per split)
train_subject_df = pd.DataFrame(
    {"participant_id": train_segment_ids.astype(str), "label": train_segment_labels.astype(str)}
).drop_duplicates(subset=["participant_id"])

val_subject_df = pd.DataFrame(
    {"participant_id": val_segment_ids.astype(str), "label": val_segment_labels.astype(str)}
).drop_duplicates(subset=["participant_id"])

test_subject_df = pd.DataFrame(
    {"participant_id": test_segment_ids.astype(str), "label": test_segment_labels.astype(str)}
).drop_duplicates(subset=["participant_id"])

subject_count_df = pd.DataFrame(
    {
        "Train": train_subject_df["label"].value_counts(),
        "Validation": val_subject_df["label"].value_counts(),
        "Test": test_subject_df["label"].value_counts(),
    }
).fillna(0).astype(int).reindex(["A", "F", "C"])

print("\nSubject-level class counts")
print(subject_count_df)

ax = subject_count_df.plot(kind="bar", figsize=(10, 5))
ax.set_title("Class counts by subjects")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Segment-level class counts
segment_count_df = pd.DataFrame(
    {
        "Train": pd.Series(train_segment_labels.astype(str)).value_counts(),
        "Validation": pd.Series(val_segment_labels.astype(str)).value_counts(),
        "Test": pd.Series(test_segment_labels.astype(str)).value_counts(),
    }
).fillna(0).astype(int).reindex(["A", "F", "C"])

print("\nSegment-level class counts")
print(segment_count_df)

ax = segment_count_df.plot(kind="bar", figsize=(10, 5))
ax.set_title("Class counts by segments")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()





#%% Optional 1: To check the IDs manually 


# Any ID cannot be occured twice, to prevent data leakage & provide subject separation
# For manual control, arrays can be saved as Excel files


desktop = Path.home() / "Desktop"    # where you want to save


# Convert to string for clean Excel output
train_ids = np.asarray(train_segment_ids).astype(str)
train_labels = np.asarray(train_segment_labels).astype(str)

val_ids = np.asarray(val_segment_ids).astype(str)
val_labels = np.asarray(val_segment_labels).astype(str)

test_ids = np.asarray(test_segment_ids).astype(str)
test_labels = np.asarray(test_segment_labels).astype(str)


# TRAIN
df_train = pd.DataFrame({"segment_id": train_ids, "segment_label": train_labels})
df_train.to_excel(desktop / "train_segment_ids_and_labels.xlsx", index=False)

# VAL
df_val = pd.DataFrame({"segment_id": val_ids, "segment_label": val_labels})
df_val.to_excel(desktop / "val_segment_ids_and_labels.xlsx", index=False)

# TEST
df_test = pd.DataFrame({"segment_id": test_ids, "segment_label": test_labels})
df_test.to_excel(desktop / "test_segment_ids_and_labels.xlsx", index=False)


print("Saved Excel files to:", desktop)



#%% 6) Data transformation for feature extraction: CWT scalograms



# CWT parameters 
scales_cwt        = np.linspace(11, 500, 60)                    
downsample_factor = 10                                         # downsample time axis: 10000 -> 1000
n_time_ds         = segment_points // downsample_factor        # time points after downsampling
n_scales_cwt      = len(scales_cwt)                            # frequency bins



def extract_cwt_scalogram(signal_1d, scales, fs=fs, downsample=downsample_factor):

    # Apply CWT on a single-channel signal 
    
    coefficients, _ = pywt.cwt(signal_1d, scales, "cmor-1.0-1.0", sampling_period=1 / fs)
    log_coefficients = 10*np.log10(np.abs(coefficients) + 1)  
    
    return log_coefficients[:, ::downsample].astype(np.float32) 




# Prepare scalogram arrays for each split

train_scalograms = np.zeros((len(train_data), n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)
val_scalograms   = np.zeros((len(val_data),   n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)
test_scalograms  = np.zeros((len(test_data),  n_channels, n_scales_cwt, n_time_ds), dtype=np.float32)


print("Computing CWT scalograms — this may take several minutes...")

for i in range(len(train_data)):
    for ch in range(n_channels):
        train_scalograms[i, ch] = extract_cwt_scalogram(train_data[i, :, ch], scales_cwt)
    if (i + 1) % 50 == 0:
        print(f"  Train: {i + 1} / {len(train_data)}")

for i in range(len(val_data)):
    for ch in range(n_channels):
        val_scalograms[i, ch] = extract_cwt_scalogram(val_data[i, :, ch], scales_cwt)

for i in range(len(test_data)):
    for ch in range(n_channels):
        test_scalograms[i, ch] = extract_cwt_scalogram(test_data[i, :, ch], scales_cwt)


print("train_scalograms shape:", train_scalograms.shape)       
print("val_scalograms shape:  ", val_scalograms.shape)
print("test_scalograms shape: ", test_scalograms.shape)




# Expected size per segment: n_channels * n_scales_cwt * n_time_ds * 4 bytes (float32)
bytes_per_segment = n_channels * n_scales_cwt * n_time_ds * 4
total_segments    = len(train_scalograms) + len(val_scalograms) + len(test_scalograms)
total_gb          = bytes_per_segment * total_segments / 1e9

print(f"\nSize estimate: {bytes_per_segment / 1e6:.1f} MB/segment × {total_segments} segments = {total_gb:.1f} GB total")
print(f"  → To reduce: lower n_scales_cwt (currently {n_scales_cwt}) or increase downsample_factor (currently {downsample_factor})")



#%% Optional 2: Load from your disk 


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
WORKSPACE_DIR = BASE_DIR.parent if BASE_DIR.name == "EEG_FTD_Alzheimer" else BASE_DIR


"""
# Optional: save to disk to skip recomputation next time
np.save(BASE_DIR / "train_scalograms.npy",     train_scalograms)
np.save(BASE_DIR / "val_scalograms.npy",       val_scalograms)
np.save(BASE_DIR / "test_scalograms.npy",      test_scalograms)
np.save(BASE_DIR / "train_segment_labels.npy", train_segment_labels)
np.save(BASE_DIR / "val_segment_labels.npy",   val_segment_labels)
np.save(BASE_DIR / "test_segment_labels.npy",  test_segment_labels)
np.save(BASE_DIR / "test_segment_ids.npy",  test_segment_ids)
"""


# Optional: load from disk (also restores label arrays needed for cell 7)
train_scalograms      = np.load(BASE_DIR / "train_scalograms.npy")
val_scalograms        = np.load(BASE_DIR / "val_scalograms.npy")
test_scalograms       = np.load(BASE_DIR / "test_scalograms.npy")
train_segment_labels  = np.load(BASE_DIR / "train_segment_labels.npy", allow_pickle=True)
val_segment_labels    = np.load(BASE_DIR / "val_segment_labels.npy",   allow_pickle=True)
test_segment_labels   = np.load(BASE_DIR / "test_segment_labels.npy",  allow_pickle=True)
test_segment_ids   = np.load(BASE_DIR / "test_segment_ids.npy",  allow_pickle=True)



#%% Optional 3: Diagnostics on labels — validate label/signal alignment 


print("DIAGNOSTIC: label-signal alignment check")


# 1) Array length consistency
assert len(train_scalograms)     == len(train_segment_labels), "MISMATCH: train scalograms vs labels"
assert len(val_scalograms)       == len(val_segment_labels),   "MISMATCH: val scalograms vs labels"
assert len(test_scalograms)      == len(test_segment_labels),  "MISMATCH: test scalograms vs labels"
print("Length check PASSED — scalograms and label arrays are same length")



# 2) Label encoding sanity: unique string labels should be A/F/C only
unique_train = set(train_segment_labels)
unique_val   = set(val_segment_labels)
unique_test  = set(test_segment_labels)
print(f"Unique labels  train: {sorted(unique_train)}")
print(f"Unique labels  val:   {sorted(unique_val)}")
print(f"Unique labels  test:  {sorted(unique_test)}")
assert unique_train <= {"A", "F", "C"}, "Unexpected label values in train"
assert unique_val   <= {"A", "F", "C"}, "Unexpected label values in val"
assert unique_test  <= {"A", "F", "C"}, "Unexpected label values in test"
print("Label value check PASSED")



# 3) Class distribution — verify fix: FTD must be present in val
from collections import Counter
c_train = Counter(train_segment_labels)
c_val   = Counter(val_segment_labels)
c_test  = Counter(test_segment_labels)
print(f"\nClass distribution (segments):")
print(f"  Train  A={c_train['A']}  F={c_train['F']}  C={c_train['C']}  "
      f"  ratios A={c_train['A']/len(train_segment_labels)*100:.1f}%"
      f" F={c_train['F']/len(train_segment_labels)*100:.1f}%"
      f" C={c_train['C']/len(train_segment_labels)*100:.1f}%")
print(f"  Val    A={c_val['A']}  F={c_val['F']}  C={c_val['C']}  "
      f"  ratios A={c_val['A']/len(val_segment_labels)*100:.1f}%"
      f" F={c_val['F']/len(val_segment_labels)*100:.1f}%"
      f" C={c_val['C']/len(val_segment_labels)*100:.1f}%")
print(f"  Test   A={c_test['A']}  F={c_test['F']}  C={c_test['C']}  "
      f"  ratios A={c_test['A']/len(test_segment_labels)*100:.1f}%"
      f" F={c_test['F']/len(test_segment_labels)*100:.1f}%"
      f" C={c_test['C']/len(test_segment_labels)*100:.1f}%")

if c_val['F'] < 5:
    print("WARNING: val has very few FTD segments — .isin() fix may NOT have been applied correctly")
else:
    print("FTD in val check PASSED")



# 4) No NaN or Inf in scalograms
for name, arr in [("train", train_scalograms), ("val", val_scalograms), ("test", test_scalograms)]:
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"  {name}: NaN={n_nan}  Inf={n_inf}")
    assert n_nan == 0, f"NaN values in {name}_scalograms"
    assert n_inf == 0, f"Inf values in {name}_scalograms"
print("NaN/Inf check PASSED")



# 5) Scalogram value range sanity (log10 compressed, values should be > 0 and < ~10)
print(f"\nScalogram value ranges:")
for name, arr in [("train", train_scalograms), ("val", val_scalograms), ("test", test_scalograms)]:
    print(f"  {name}: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")



# 6) Spot-check: first segment of each class in train — label must match segment_id's known class
label_map_check = {"A": 0, "F": 1, "C": 2}
train_labels_check = np.array([label_map_check[l] for l in train_segment_labels], dtype=np.int64)
val_labels_check   = np.array([label_map_check[l] for l in val_segment_labels],   dtype=np.int64)
test_labels_check  = np.array([label_map_check[l] for l in test_segment_labels],  dtype=np.int64)

print(f"\nFirst 10 train labels (string): {list(train_segment_labels[:10])}")
print(f"First 10 train labels (int):    {list(train_labels_check[:10])}")
print(f"First 10 train subject IDs:     {list(train_segment_ids[:10])}")



# 7) No subject appears in more than one split
train_id_set = set(train_segment_ids)
val_id_set   = set(val_segment_ids)
test_id_set  = set(test_segment_ids)
tv = train_id_set & val_id_set
tt = train_id_set & test_id_set
vt = val_id_set   & test_id_set
print(f"\nData leakage check:")
print(f"  Train ∩ Val:  {tv if tv else 'none'}")
print(f"  Train ∩ Test: {tt if tt else 'none'}")
print(f"  Val ∩ Test:   {vt if vt else 'none'}")
assert not tv and not tt and not vt, "DATA LEAKAGE DETECTED"
print("Data leakage check PASSED")

print("\nAll diagnostics PASSED — proceed to training")




#%% 6.2) Binary label arrays for one-vs-rest


# AD vs rest  (AD=1, FTD=0, HC=0)
train_labels_bin_ad = (train_segment_labels == "A").astype(np.int64)
val_labels_bin_ad   = (val_segment_labels   == "A").astype(np.int64)
test_labels_bin_ad  = (test_segment_labels  == "A").astype(np.int64)

# FTD vs rest  (FTD=1, AD=0, HC=0)
train_labels_bin_ftd = (train_segment_labels == "F").astype(np.int64)
val_labels_bin_ftd   = (val_segment_labels   == "F").astype(np.int64)
test_labels_bin_ftd  = (test_segment_labels  == "F").astype(np.int64)

# HC vs rest  (HC=1, AD=0, FTD=0)
train_labels_bin_hc = (train_segment_labels == "C").astype(np.int64)
val_labels_bin_hc   = (val_segment_labels   == "C").astype(np.int64)
test_labels_bin_hc  = (test_segment_labels  == "C").astype(np.int64)

# Multi-class integer labels  (A=0, F=1, C=2) — for combined evaluation in cells 8 and 9
label_map        = {"A": 0, "F": 1, "C": 2}
train_labels_mc  = np.array([label_map[l] for l in train_segment_labels], dtype=np.int64)
val_labels_mc    = np.array([label_map[l] for l in val_segment_labels],   dtype=np.int64)
test_labels_mc   = np.array([label_map[l] for l in test_segment_labels],  dtype=np.int64)





#%% 7) Model setup


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns




print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")




# ----Hyperparameters  (shared across all three binary models)
lr           = 0.0001
batch_size   = 8
epochs       = 200
dropout      = 0.20
weight_decay = 0.0005
label_smooth = 0.01
grad_clip    = 2.0




# Scalogram tensors  (shared; DataLoaders are defined per binary model in cells 7.1–7.3)
train_tensor = torch.tensor(train_scalograms, dtype=torch.float32)
val_tensor   = torch.tensor(val_scalograms,   dtype=torch.float32)
test_tensor  = torch.tensor(test_scalograms,  dtype=torch.float32)




# ---- Model definition: YOLO26-cls adapted for 19-channel EEG scalograms


model_weights_cls = "yolo26m-cls.pt"




class YOLO26CLS(nn.Module):
    """
    YOLO26-cls backbone adapted for 19-channel EEG scalogram input: [B, 19, 60, 1000].

    Main correction:
    The first convolution is initialized from pretrained RGB weights instead of
    being randomly initialized.

    RGB pretrained conv:
        [out_channels, 3, kh, kw]

    EEG adapted conv:
        [out_channels, 19, kh, kw]
    """

    def __init__(self, weights=model_weights_cls, num_classes=3, dropout=0.20):
        super().__init__()

        _yolo = YOLO(weights)
        self.backbone = _yolo.model.float()

        
        # Patch first conv: 3 RGB channels -> 19 EEG channels
        # using pretrained RGB weights
        
        first_conv = self.backbone.model[0].conv

        old_weight = first_conv.weight.data.clone()
        old_bias_exists = first_conv.bias is not None

        new_conv = nn.Conv2d(
            in_channels  = 19,
            out_channels = first_conv.out_channels,
            kernel_size  = first_conv.kernel_size,
            stride       = first_conv.stride,
            padding      = first_conv.padding,
            dilation     = first_conv.dilation,
            groups       = first_conv.groups,
            bias         = old_bias_exists,
            padding_mode = first_conv.padding_mode,
        )

        with torch.no_grad():
            # old_weight: [out_channels, 3, kh, kw]

            # Average pretrained RGB filters:
            # [out_channels, 3, kh, kw] -> [out_channels, 1, kh, kw]
            rgb_mean_weight = old_weight.mean(dim=1, keepdim=True)

            # Repeat averaged filter to 19 EEG channels:
            # [out_channels, 1, kh, kw] -> [out_channels, 19, kh, kw]
            new_weight = rgb_mean_weight.repeat(1, 19, 1, 1)

            # Preserve activation scale.
            # Original conv sums 3 channels; new conv sums 19 channels.
            new_weight = new_weight * (3.0 / 19.0)

            new_conv.weight.copy_(new_weight)

            if old_bias_exists:
                new_conv.bias.copy_(first_conv.bias.data)

        self.backbone.model[0].conv = new_conv

        
        # Patch classifier head: original classes -> 3 EEG classes
        
        classify_head = self.backbone.model[-1]

        in_feats = classify_head.linear.in_features
        classify_head.linear = nn.Linear(in_feats, num_classes)

        # Adjust classification dropout if available
        if hasattr(classify_head, "drop"):
            classify_head.drop.p = dropout

        # Initialize only the new classifier layer
        nn.init.kaiming_normal_(
            classify_head.linear.weight,
            mode="fan_in",
            nonlinearity="relu"
        )
        nn.init.constant_(classify_head.linear.bias, 0)

        
        # Force raw logits output.
        # Ultralytics classification models may return softmax in eval mode.
        # CrossEntropyLoss requires raw logits.
        
        _h = classify_head

        def _logits_forward(x):
            if isinstance(x, list):
                x = torch.cat(x, 1)
            return _h.linear(_h.drop(_h.pool(_h.conv(x)).flatten(1)))

        classify_head.forward = _logits_forward

        del _yolo

    def forward(self, x):
        out = self.backbone(x)

        # Some Ultralytics versions may return tuple/list.
        # Extract logits safely.
        if isinstance(out, (tuple, list)):
            out = out[-1]

        return out






#%% 7.1) Binary model: AD vs rest  (model_ad)


train_loader_ad = DataLoader(
    TensorDataset(train_tensor, torch.tensor(train_labels_bin_ad)),
    batch_size=batch_size,
    shuffle=True
)
val_loader_ad = DataLoader(
    TensorDataset(val_tensor, torch.tensor(val_labels_bin_ad)),
    batch_size=batch_size,
    shuffle=False
)


model_ad = YOLO26CLS(weights=model_weights_cls, num_classes=2, dropout=dropout).to(device)


counts_ad  = np.bincount(train_labels_bin_ad, minlength=2)
weights_ad = len(train_labels_bin_ad) / (2 * counts_ad)
weights_ad = torch.tensor(weights_ad, dtype=torch.float32).to(device)


print("model_ad — class counts  (0=rest, 1=AD):", counts_ad)
print("model_ad — class weights:", weights_ad.detach().cpu().numpy())

criterion_ad = nn.CrossEntropyLoss(weight=weights_ad, label_smoothing=label_smooth)
optimizer_ad = optim.AdamW(model_ad.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_ad = optim.lr_scheduler.CosineAnnealingLR(optimizer_ad, T_max=epochs)


train_losses_ad     = []
val_losses_ad       = []
train_accuracies_ad = []
val_accuracies_ad   = []


for epoch in range(epochs):

    model_ad.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    for inputs, labels in train_loader_ad:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_ad.zero_grad()
        outputs = model_ad(inputs)
        loss    = criterion_ad(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ad.parameters(), max_norm=grad_clip)
        optimizer_ad.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_loss     = running_loss / len(train_loader_ad)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses_ad.append(train_loss)
    train_accuracies_ad.append(train_accuracy)

    model_ad.eval()
    val_running_loss = 0.0
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs, labels in val_loader_ad:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ad(inputs)
            loss    = criterion_ad(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss     = val_running_loss / len(val_loader_ad)
    val_accuracy = accuracy_score(y_true_val, y_pred_val) * 100
    val_losses_ad.append(val_loss)
    val_accuracies_ad.append(val_accuracy)

    scheduler_ad.step()

    print(f"Epoch {epoch+1}/{epochs}  |  Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.2f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_accuracy:.2f}%")


# Validation evaluation
model_ad.eval()
y_true_val_ad  = []
y_pred_val_ad  = []
y_proba_val_ad = []

with torch.no_grad():
    for inputs, labels in val_loader_ad:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_ad(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_val_ad.extend(labels.cpu().numpy())
        y_pred_val_ad.extend(predicted.cpu().numpy())
        y_proba_val_ad.extend(proba.cpu().numpy())

y_true_val_ad  = np.array(y_true_val_ad)
y_pred_val_ad  = np.array(y_pred_val_ad)
y_proba_val_ad = np.array(y_proba_val_ad)

acc_val  = accuracy_score(y_true_val_ad, y_pred_val_ad)
f1_val   = f1_score(y_true_val_ad, y_pred_val_ad, average="binary")
prec_val = precision_score(y_true_val_ad, y_pred_val_ad, average="binary", zero_division=0)
rec_val  = recall_score(y_true_val_ad, y_pred_val_ad, average="binary")

print(f"\nmodel_ad — Val Accuracy:  {acc_val  * 100:.2f}%")
print(f"model_ad — Val F1:        {f1_val   * 100:.2f}%")
print(f"model_ad — Val Precision: {prec_val * 100:.2f}%")
print(f"model_ad — Val Recall:    {rec_val  * 100:.2f}%")

cm_val_ad = confusion_matrix(y_true_val_ad, y_pred_val_ad)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_val_ad, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "AD"], yticklabels=["Rest", "AD"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation | model_ad")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_ad, label="Train")
plt.plot(val_losses_ad,   label="Validation")
plt.title("Loss over epochs | model_ad")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies_ad, label="Train")
plt.plot(val_accuracies_ad,   label="Validation")
plt.title("Accuracy over epochs | model_ad")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()

fpr_ad_val, tpr_ad_val, _ = roc_curve(y_true_val_ad, y_proba_val_ad[:, 1])
auc_ad_val = auc(fpr_ad_val, tpr_ad_val)
plt.figure(figsize=(8, 6))
plt.plot(fpr_ad_val, tpr_ad_val, label=f"AD vs Rest  AUC = {auc_ad_val:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation | model_ad")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Binary test evaluation
test_loader_ad = DataLoader(
    TensorDataset(test_tensor, torch.tensor(test_labels_bin_ad)),
    batch_size=batch_size, shuffle=False
)
model_ad.eval()
y_true_test_ad  = []
y_pred_test_ad  = []
y_proba_test_ad = []

with torch.no_grad():
    for inputs, labels in test_loader_ad:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_ad(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_test_ad.extend(labels.cpu().numpy())
        y_pred_test_ad.extend(predicted.cpu().numpy())
        y_proba_test_ad.extend(proba.cpu().numpy())

y_true_test_ad  = np.array(y_true_test_ad)
y_pred_test_ad  = np.array(y_pred_test_ad)
y_proba_test_ad = np.array(y_proba_test_ad)

acc_test_ad  = accuracy_score(y_true_test_ad, y_pred_test_ad)
f1_test_ad   = f1_score(y_true_test_ad, y_pred_test_ad, average="binary")
prec_test_ad = precision_score(y_true_test_ad, y_pred_test_ad, average="binary", zero_division=0)
rec_test_ad  = recall_score(y_true_test_ad, y_pred_test_ad, average="binary")

print(f"\nmodel_ad — Test Accuracy:  {acc_test_ad  * 100:.2f}%")
print(f"model_ad — Test F1:        {f1_test_ad   * 100:.2f}%")
print(f"model_ad — Test Precision: {prec_test_ad * 100:.2f}%")
print(f"model_ad — Test Recall:    {rec_test_ad  * 100:.2f}%")

cm_test_ad = confusion_matrix(y_true_test_ad, y_pred_test_ad)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_ad, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "AD"], yticklabels=["Rest", "AD"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test | model_ad")
plt.tight_layout()
plt.show()




#%% 7.2) Binary model: FTD vs rest  (model_ftd)


train_loader_ftd = DataLoader(
    TensorDataset(train_tensor, torch.tensor(train_labels_bin_ftd)),
    batch_size=batch_size,
    shuffle=True
)
val_loader_ftd = DataLoader(
    TensorDataset(val_tensor, torch.tensor(val_labels_bin_ftd)),
    batch_size=batch_size,
    shuffle=False
)


model_ftd = YOLO26CLS(weights=model_weights_cls, num_classes=2, dropout=dropout).to(device)


counts_ftd  = np.bincount(train_labels_bin_ftd, minlength=2)
weights_ftd = len(train_labels_bin_ftd) / (2 * counts_ftd)
weights_ftd = torch.tensor(weights_ftd, dtype=torch.float32).to(device)




print("model_ftd — class counts  (0=rest, 1=FTD):", counts_ftd)
print("model_ftd — class weights:", weights_ftd.detach().cpu().numpy())

criterion_ftd = nn.CrossEntropyLoss(weight=weights_ftd, label_smoothing=label_smooth)
optimizer_ftd = optim.AdamW(model_ftd.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_ftd = optim.lr_scheduler.CosineAnnealingLR(optimizer_ftd, T_max=epochs)


train_losses_ftd     = []
val_losses_ftd       = []
train_accuracies_ftd = []
val_accuracies_ftd   = []


for epoch in range(epochs):

    model_ftd.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    for inputs, labels in train_loader_ftd:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_ftd.zero_grad()
        outputs = model_ftd(inputs)
        loss    = criterion_ftd(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ftd.parameters(), max_norm=grad_clip)
        optimizer_ftd.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_loss     = running_loss / len(train_loader_ftd)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses_ftd.append(train_loss)
    train_accuracies_ftd.append(train_accuracy)

    model_ftd.eval()
    val_running_loss = 0.0
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs, labels in val_loader_ftd:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ftd(inputs)
            loss    = criterion_ftd(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss     = val_running_loss / len(val_loader_ftd)
    val_accuracy = accuracy_score(y_true_val, y_pred_val) * 100
    val_losses_ftd.append(val_loss)
    val_accuracies_ftd.append(val_accuracy)

    scheduler_ftd.step()

    print(f"Epoch {epoch+1}/{epochs}  |  Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.2f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_accuracy:.2f}%")


# Validation evaluation
model_ftd.eval()
y_true_val_ftd  = []
y_pred_val_ftd  = []
y_proba_val_ftd = []

with torch.no_grad():
    for inputs, labels in val_loader_ftd:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_ftd(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_val_ftd.extend(labels.cpu().numpy())
        y_pred_val_ftd.extend(predicted.cpu().numpy())
        y_proba_val_ftd.extend(proba.cpu().numpy())

y_true_val_ftd  = np.array(y_true_val_ftd)
y_pred_val_ftd  = np.array(y_pred_val_ftd)
y_proba_val_ftd = np.array(y_proba_val_ftd)

acc_val  = accuracy_score(y_true_val_ftd, y_pred_val_ftd)
f1_val   = f1_score(y_true_val_ftd, y_pred_val_ftd, average="binary")
prec_val = precision_score(y_true_val_ftd, y_pred_val_ftd, average="binary", zero_division=0)
rec_val  = recall_score(y_true_val_ftd, y_pred_val_ftd, average="binary")

print(f"\nmodel_ftd — Val Accuracy:  {acc_val  * 100:.2f}%")
print(f"model_ftd — Val F1:        {f1_val   * 100:.2f}%")
print(f"model_ftd — Val Precision: {prec_val * 100:.2f}%")
print(f"model_ftd — Val Recall:    {rec_val  * 100:.2f}%")

cm_val_ftd = confusion_matrix(y_true_val_ftd, y_pred_val_ftd)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_val_ftd, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "FTD"], yticklabels=["Rest", "FTD"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation | model_ftd")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_ftd, label="Train")
plt.plot(val_losses_ftd,   label="Validation")
plt.title("Loss over epochs | model_ftd")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies_ftd, label="Train")
plt.plot(val_accuracies_ftd,   label="Validation")
plt.title("Accuracy over epochs | model_ftd")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()

fpr_ftd_val, tpr_ftd_val, _ = roc_curve(y_true_val_ftd, y_proba_val_ftd[:, 1])
auc_ftd_val = auc(fpr_ftd_val, tpr_ftd_val)
plt.figure(figsize=(8, 6))
plt.plot(fpr_ftd_val, tpr_ftd_val, label=f"FTD vs Rest  AUC = {auc_ftd_val:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation | model_ftd")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Binary test evaluation
test_loader_ftd = DataLoader(
    TensorDataset(test_tensor, torch.tensor(test_labels_bin_ftd)),
    batch_size=batch_size, shuffle=False
)
model_ftd.eval()
y_true_test_ftd  = []
y_pred_test_ftd  = []
y_proba_test_ftd = []

with torch.no_grad():
    for inputs, labels in test_loader_ftd:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_ftd(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_test_ftd.extend(labels.cpu().numpy())
        y_pred_test_ftd.extend(predicted.cpu().numpy())
        y_proba_test_ftd.extend(proba.cpu().numpy())

y_true_test_ftd  = np.array(y_true_test_ftd)
y_pred_test_ftd  = np.array(y_pred_test_ftd)
y_proba_test_ftd = np.array(y_proba_test_ftd)

acc_test_ftd  = accuracy_score(y_true_test_ftd, y_pred_test_ftd)
f1_test_ftd   = f1_score(y_true_test_ftd, y_pred_test_ftd, average="binary")
prec_test_ftd = precision_score(y_true_test_ftd, y_pred_test_ftd, average="binary", zero_division=0)
rec_test_ftd  = recall_score(y_true_test_ftd, y_pred_test_ftd, average="binary")

print(f"\nmodel_ftd — Test Accuracy:  {acc_test_ftd  * 100:.2f}%")
print(f"model_ftd — Test F1:        {f1_test_ftd   * 100:.2f}%")
print(f"model_ftd — Test Precision: {prec_test_ftd * 100:.2f}%")
print(f"model_ftd — Test Recall:    {rec_test_ftd  * 100:.2f}%")

cm_test_ftd = confusion_matrix(y_true_test_ftd, y_pred_test_ftd)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_ftd, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "FTD"], yticklabels=["Rest", "FTD"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test | model_ftd")
plt.tight_layout()
plt.show()




#%% 7.3) Binary model: HC vs rest  (model_hc)


train_loader_hc = DataLoader(
    TensorDataset(train_tensor, torch.tensor(train_labels_bin_hc)),
    batch_size=batch_size,
    shuffle=True
)
val_loader_hc = DataLoader(
    TensorDataset(val_tensor, torch.tensor(val_labels_bin_hc)),
    batch_size=batch_size,
    shuffle=False
)


model_hc = YOLO26CLS(weights=model_weights_cls, num_classes=2, dropout=dropout).to(device)


counts_hc  = np.bincount(train_labels_bin_hc, minlength=2)
weights_hc = len(train_labels_bin_hc) / (2 * counts_hc)
weights_hc = torch.tensor(weights_hc, dtype=torch.float32).to(device)




print("model_hc — class counts  (0=rest, 1=HC):", counts_hc)
print("model_hc — class weights:", weights_hc.detach().cpu().numpy())

criterion_hc = nn.CrossEntropyLoss(weight=weights_hc, label_smoothing=label_smooth)
optimizer_hc = optim.AdamW(model_hc.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_hc = optim.lr_scheduler.CosineAnnealingLR(optimizer_hc, T_max=epochs)


train_losses_hc     = []
val_losses_hc       = []
train_accuracies_hc = []
val_accuracies_hc   = []


for epoch in range(epochs):

    model_hc.train()
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    for inputs, labels in train_loader_hc:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_hc.zero_grad()
        outputs = model_hc(inputs)
        loss    = criterion_hc(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_hc.parameters(), max_norm=grad_clip)
        optimizer_hc.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_loss     = running_loss / len(train_loader_hc)
    train_accuracy = accuracy_score(y_true_train, y_pred_train) * 100
    train_losses_hc.append(train_loss)
    train_accuracies_hc.append(train_accuracy)

    model_hc.eval()
    val_running_loss = 0.0
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs, labels in val_loader_hc:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_hc(inputs)
            loss    = criterion_hc(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss     = val_running_loss / len(val_loader_hc)
    val_accuracy = accuracy_score(y_true_val, y_pred_val) * 100
    val_losses_hc.append(val_loss)
    val_accuracies_hc.append(val_accuracy)

    scheduler_hc.step()

    print(f"Epoch {epoch+1}/{epochs}  |  Train Loss: {train_loss:.4f}  Train Acc: {train_accuracy:.2f}%  |  Val Loss: {val_loss:.4f}  Val Acc: {val_accuracy:.2f}%")


# Validation evaluation
model_hc.eval()
y_true_val_hc  = []
y_pred_val_hc  = []
y_proba_val_hc = []

with torch.no_grad():
    for inputs, labels in val_loader_hc:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_hc(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_val_hc.extend(labels.cpu().numpy())
        y_pred_val_hc.extend(predicted.cpu().numpy())
        y_proba_val_hc.extend(proba.cpu().numpy())

y_true_val_hc  = np.array(y_true_val_hc)
y_pred_val_hc  = np.array(y_pred_val_hc)
y_proba_val_hc = np.array(y_proba_val_hc)

acc_val  = accuracy_score(y_true_val_hc, y_pred_val_hc)
f1_val   = f1_score(y_true_val_hc, y_pred_val_hc, average="binary")
prec_val = precision_score(y_true_val_hc, y_pred_val_hc, average="binary", zero_division=0)
rec_val  = recall_score(y_true_val_hc, y_pred_val_hc, average="binary")

print(f"\nmodel_hc — Val Accuracy:  {acc_val  * 100:.2f}%")
print(f"model_hc — Val F1:        {f1_val   * 100:.2f}%")
print(f"model_hc — Val Precision: {prec_val * 100:.2f}%")
print(f"model_hc — Val Recall:    {rec_val  * 100:.2f}%")

cm_val_hc = confusion_matrix(y_true_val_hc, y_pred_val_hc)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_val_hc, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "HC"], yticklabels=["Rest", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation | model_hc")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_hc, label="Train")
plt.plot(val_losses_hc,   label="Validation")
plt.title("Loss over epochs | model_hc")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies_hc, label="Train")
plt.plot(val_accuracies_hc,   label="Validation")
plt.title("Accuracy over epochs | model_hc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()

fpr_hc_val, tpr_hc_val, _ = roc_curve(y_true_val_hc, y_proba_val_hc[:, 1])
auc_hc_val = auc(fpr_hc_val, tpr_hc_val)
plt.figure(figsize=(8, 6))
plt.plot(fpr_hc_val, tpr_hc_val, label=f"HC vs Rest  AUC = {auc_hc_val:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation | model_hc")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Binary test evaluation
test_loader_hc = DataLoader(
    TensorDataset(test_tensor, torch.tensor(test_labels_bin_hc)),
    batch_size=batch_size, shuffle=False
)
model_hc.eval()
y_true_test_hc  = []
y_pred_test_hc  = []
y_proba_test_hc = []

with torch.no_grad():
    for inputs, labels in test_loader_hc:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_hc(inputs)
        proba   = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        y_true_test_hc.extend(labels.cpu().numpy())
        y_pred_test_hc.extend(predicted.cpu().numpy())
        y_proba_test_hc.extend(proba.cpu().numpy())

y_true_test_hc  = np.array(y_true_test_hc)
y_pred_test_hc  = np.array(y_pred_test_hc)
y_proba_test_hc = np.array(y_proba_test_hc)

acc_test_hc  = accuracy_score(y_true_test_hc, y_pred_test_hc)
f1_test_hc   = f1_score(y_true_test_hc, y_pred_test_hc, average="binary")
prec_test_hc = precision_score(y_true_test_hc, y_pred_test_hc, average="binary", zero_division=0)
rec_test_hc  = recall_score(y_true_test_hc, y_pred_test_hc, average="binary")

print(f"\nmodel_hc — Test Accuracy:  {acc_test_hc  * 100:.2f}%")
print(f"model_hc — Test F1:        {f1_test_hc   * 100:.2f}%")
print(f"model_hc — Test Precision: {prec_test_hc * 100:.2f}%")
print(f"model_hc — Test Recall:    {rec_test_hc  * 100:.2f}%")

cm_test_hc = confusion_matrix(y_true_test_hc, y_pred_test_hc)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_test_hc, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rest", "HC"], yticklabels=["Rest", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test | model_hc")
plt.tight_layout()
plt.show()





#%% 7.4) Validation-based threshold selection for combined predictions


# Stack validation probabilities from the three binary models
# Columns: [P(AD), P(FTD), P(HC)]
y_proba_val_mc = np.stack(
    [
        y_proba_val_ad[:, 1],
        y_proba_val_ftd[:, 1],
        y_proba_val_hc[:, 1],
    ],
    axis=1
)

y_true_val_mc = val_labels_mc


def predict_with_thresholds(y_proba, thresholds):
    adjusted_scores = y_proba - thresholds.reshape(1, 3)
    return np.argmax(adjusted_scores, axis=1)


# Coarser grid: much faster than 0.01 step
# 0.05 step gives 19 × 19 × 19 = 6859 combinations
threshold_values = np.arange(0.05, 0.96, 0.05)

best_thresholds = None
best_val_score = -1


for th_ad in threshold_values:
    for th_ftd in threshold_values:
        for th_hc in threshold_values:

            thresholds = np.array([th_ad, th_ftd, th_hc], dtype=np.float32)

            y_pred_val_mc = predict_with_thresholds(
                y_proba_val_mc,
                thresholds
            )

            # Use only weighted F1 inside the loop.
            # This keeps the search loyal to your current final evaluation metric.
            f1_val_thr = f1_score(
                y_true_val_mc,
                y_pred_val_mc,
                average="weighted",
                zero_division=0
            )

            if f1_val_thr > best_val_score:
                best_val_score = f1_val_thr
                best_thresholds = thresholds.copy()


# After selecting the best thresholds, calculate full validation metrics once
y_pred_val_best = predict_with_thresholds(
    y_proba_val_mc,
    best_thresholds
)

best_val_acc = accuracy_score(y_true_val_mc, y_pred_val_best)

best_val_f1 = f1_score(
    y_true_val_mc,
    y_pred_val_best,
    average="weighted",
    zero_division=0
)

best_val_precision = precision_score(
    y_true_val_mc,
    y_pred_val_best,
    average="weighted",
    zero_division=0
)

best_val_recall = recall_score(
    y_true_val_mc,
    y_pred_val_best,
    average="weighted"
)


print("\nBest validation-based thresholds")
print("--------------------------------")
print(f"AD threshold:  {best_thresholds[0]:.2f}")
print(f"FTD threshold: {best_thresholds[1]:.2f}")
print(f"HC threshold:  {best_thresholds[2]:.2f}")

print("\nValidation performance with selected thresholds")
print("------------------------------------------------")
print(f"Val Accuracy:  {best_val_acc * 100:.2f}%")
print(f"Val F1:        {best_val_f1 * 100:.2f}%")
print(f"Val Precision: {best_val_precision * 100:.2f}%")
print(f"Val Recall:    {best_val_recall * 100:.2f}%")


cm_val_thresholded = confusion_matrix(y_true_val_mc, y_pred_val_best)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_val_thresholded,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["AD", "FTD", "HC"],
    yticklabels=["AD", "FTD", "HC"]
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Validation (Thresholded Combined OvR)")
plt.tight_layout()
plt.show()





#%% Optional 4: Diagnostic quick model for a health check
# Runs a binary health check using model_ad's setup as a reference.
# Requires cells 7 and 7.1 to have been executed first.


import copy
from collections import Counter

def quick_model_health_check(ModelClass, model_kwargs=None, tiny_n=16, lr_diag=1e-4):
    if model_kwargs is None:
        model_kwargs = {}

    print("=" * 70)
    print("QUICK MODEL HEALTH CHECK")
    print("=" * 70)


    # 1) Tiny real-label memorization

    print("\n1) Tiny real-label memorization")

    x_tiny = train_tensor[:tiny_n].to(device)
    y_tiny = torch.tensor(train_labels_bin_ad[:tiny_n], dtype=torch.long).to(device)

    print("Tiny labels:", y_tiny.detach().cpu().numpy())
    print("Tiny label counts [0=rest, 1=AD]:", np.bincount(y_tiny.detach().cpu().numpy(), minlength=2))

    model_tiny = ModelClass(**model_kwargs).to(device)
    criterion_diag = nn.CrossEntropyLoss()
    optimizer_tiny = optim.Adam(model_tiny.parameters(), lr=lr_diag)

    tiny_success = False

    for epoch in range(100):
        model_tiny.train()
        optimizer_tiny.zero_grad()
        outputs = model_tiny(x_tiny)
        loss = criterion_diag(outputs, y_tiny)
        loss.backward()
        optimizer_tiny.step()
        preds = outputs.argmax(dim=1)
        acc = (preds == y_tiny).float().mean().item() * 100
        if (epoch + 1) % 10 == 0 or acc == 100:
            print(f"Epoch {epoch+1:03d} | Loss={loss.item():.4f} | Acc={acc:.2f}%")
        if acc == 100:
            tiny_success = True
            break


    # 2) Tiny random-label memorization

    print("\n2) Tiny random-label memorization")

    y_random = torch.randint(0, 2, size=(tiny_n,), dtype=torch.long).to(device)

    print("Random labels:", y_random.detach().cpu().numpy())
    print("Random label counts [0=rest, 1=AD]:", np.bincount(y_random.detach().cpu().numpy(), minlength=2))

    model_random = ModelClass(**model_kwargs).to(device)
    optimizer_random = optim.Adam(model_random.parameters(), lr=lr_diag)

    random_success = False

    for epoch in range(150):
        model_random.train()
        optimizer_random.zero_grad()
        outputs = model_random(x_tiny)
        loss = criterion_diag(outputs, y_random)
        loss.backward()
        optimizer_random.step()
        preds = outputs.argmax(dim=1)
        acc = (preds == y_random).float().mean().item() * 100
        if (epoch + 1) % 15 == 0 or acc == 100:
            print(f"Epoch {epoch+1:03d} | Loss={loss.item():.4f} | Acc={acc:.2f}%")
        if acc == 100:
            random_success = True
            break


    # 3) Initial batch logits check

    print("\n3) Initial batch logits/probabilities")

    model_init = ModelClass(**model_kwargs).to(device)
    model_init.train()

    inputs, labels = next(iter(train_loader_ad))
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model_init(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    print("Input min/max/mean/std:",
          inputs.min().item(), inputs.max().item(),
          inputs.mean().item(), inputs.std().item())
    print("Logits min/max/mean/std:",
          logits.min().item(), logits.max().item(),
          logits.mean().item(), logits.std().item())
    print("Labels:", labels.detach().cpu().numpy())
    print("Initial preds:", preds.detach().cpu().numpy())
    print("Initial mean probs:", probs.mean(dim=0).detach().cpu().numpy())
    print("Initial prob std:", probs.std(dim=0).detach().cpu().numpy())


    # 4) One epoch full-train behavior

    print("\n4) One-epoch full-train behavior")

    model_one = ModelClass(**model_kwargs).to(device)

    counts_diag  = np.bincount(train_labels_bin_ad, minlength=2)
    weights_diag = len(train_labels_bin_ad) / (2 * counts_diag)
    weights_diag = torch.tensor(weights_diag, dtype=torch.float32).to(device)

    criterion_weighted = nn.CrossEntropyLoss(weight=weights_diag)
    optimizer_one = optim.Adam(model_one.parameters(), lr=lr_diag)

    model_one.train()
    y_true_one = []
    y_pred_one = []
    running_loss = 0.0

    for inputs, labels in train_loader_ad:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer_one.zero_grad()
        outputs = model_one(inputs)
        loss = criterion_weighted(outputs, labels)
        loss.backward()
        optimizer_one.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        y_true_one.extend(labels.detach().cpu().numpy())
        y_pred_one.extend(preds.detach().cpu().numpy())

    one_epoch_acc = accuracy_score(y_true_one, y_pred_one) * 100
    pred_counts = np.bincount(np.array(y_pred_one), minlength=2)

    print(f"One-epoch train loss: {running_loss / len(train_loader_ad):.4f}")
    print(f"One-epoch train acc:  {one_epoch_acc:.2f}%")
    print("One-epoch pred counts [0=rest, 1=AD]:", pred_counts.tolist())


    # 5) Summary

    print("\n5) Summary")
    print("Tiny real-label memorization:", "PASS" if tiny_success else "FAIL")
    print("Tiny random-label memorization:", "PASS" if random_success else "FAIL")
    if pred_counts.min() == 0:
        print("WARNING: one-epoch model ignores at least one class.")
    else:
        print("Prediction diversity check: PASS")


quick_model_health_check(
    YOLO26CLS,
    model_kwargs={"num_classes": 2},
    tiny_n=16,
    lr_diag=1e-4
)




#%% 8) Combined three-class evaluation — segment level


# Each binary model outputs P(positive class | segment).
# Stacking [P(AD), P(FTD), P(HC)] and taking argmax gives a 3-class prediction.

model_ad.eval()
model_ftd.eval()
model_hc.eval()

proba_ad_test  = []  # P(AD)  from model_ad  — index 1 of binary softmax
proba_ftd_test = []  # P(FTD) from model_ftd
proba_hc_test  = []  # P(HC)  from model_hc
y_true_test_mc = []

test_loader_mc = DataLoader(
    TensorDataset(test_tensor, torch.tensor(test_labels_mc)),
    batch_size=batch_size,
    shuffle=False
)

with torch.no_grad():
    for inputs, labels in test_loader_mc:
        inputs = inputs.to(device)

        p_ad  = torch.softmax(model_ad(inputs),  dim=1)[:, 1]
        p_ftd = torch.softmax(model_ftd(inputs), dim=1)[:, 1]
        p_hc  = torch.softmax(model_hc(inputs),  dim=1)[:, 1]

        proba_ad_test.extend(p_ad.cpu().numpy())
        proba_ftd_test.extend(p_ftd.cpu().numpy())
        proba_hc_test.extend(p_hc.cpu().numpy())
        y_true_test_mc.extend(labels.cpu().numpy())

y_true_test_mc  = np.array(y_true_test_mc)
proba_ad_test   = np.array(proba_ad_test)
proba_ftd_test  = np.array(proba_ftd_test)
proba_hc_test   = np.array(proba_hc_test)



# y_proba_test_mc: (n_segments, 3)  —  columns: [P(AD), P(FTD), P(HC)]
y_proba_test_mc = np.stack([proba_ad_test, proba_ftd_test, proba_hc_test], axis=1)

# Apply validation-selected thresholds
y_pred_test_mc = predict_with_thresholds(
    y_proba_test_mc,
    best_thresholds
)



acc_test  = accuracy_score(y_true_test_mc, y_pred_test_mc)
f1_test   = f1_score(y_true_test_mc, y_pred_test_mc, average="weighted")
prec_test = precision_score(y_true_test_mc, y_pred_test_mc, average="weighted", zero_division=0)
rec_test  = recall_score(y_true_test_mc, y_pred_test_mc, average="weighted")

print(f"\nTest Accuracy:  {acc_test  * 100:.2f}%")
print(f"Test F1:        {f1_test   * 100:.2f}%")
print(f"Test Precision: {prec_test * 100:.2f}%")
print(f"Test Recall:    {rec_test  * 100:.2f}%")


cm_test = confusion_matrix(y_true_test_mc, y_pred_test_mc)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=["AD", "FTD", "HC"],
            yticklabels=["AD", "FTD", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test (Combined OvR)")
plt.tight_layout()
plt.show()


class_names_roc    = ["AD (A)", "FTD (F)", "HC (C)"]
y_true_test_mc_bin = label_binarize(y_true_test_mc, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_test_mc_bin[:, i], y_proba_test_mc[:, i])
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_roc[i]}  AUC = {auc_score:.3f}")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test (Combined OvR)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()




#%% 9) Combined three-class evaluation — subject level


# Safety check
print("len(y_true_test_mc): ", len(y_true_test_mc))
print("len(y_pred_test_mc): ", len(y_pred_test_mc))
print("len(y_proba_test_mc):", len(y_proba_test_mc))
print("len(test_segment_ids):", len(test_segment_ids))

assert len(y_true_test_mc)  == len(test_segment_ids), "Mismatch: y_true_test_mc vs test_segment_ids"
assert len(y_pred_test_mc)  == len(test_segment_ids), "Mismatch: y_pred_test_mc vs test_segment_ids"
assert len(y_proba_test_mc) == len(test_segment_ids), "Mismatch: y_proba_test_mc vs test_segment_ids"


# Aggregate to subject level by mean probability across segments
subject_proba_dict = {}
subject_true_dict  = {}

for i in range(len(test_segment_ids)):
    sid = str(test_segment_ids[i])
    if sid not in subject_proba_dict:
        subject_proba_dict[sid] = []
    subject_proba_dict[sid].append(y_proba_test_mc[i])
    if sid not in subject_true_dict:
        subject_true_dict[sid] = int(y_true_test_mc[i])

subject_ids_sorted = sorted(subject_proba_dict.keys())

y_true_test_subj  = []
y_pred_test_subj  = []
y_proba_test_subj = []



for sid in subject_ids_sorted:
    proba_array = np.array(subject_proba_dict[sid])   # (n_segments_subj, 3)
    mean_proba  = proba_array.mean(axis=0)             # (3,)

    # Apply the same validation-selected thresholds at subject level
    adjusted_mean_proba = mean_proba - best_thresholds
    pred_class = int(np.argmax(adjusted_mean_proba))

    y_true_test_subj.append(subject_true_dict[sid])
    y_pred_test_subj.append(pred_class)
    y_proba_test_subj.append(mean_proba)


y_true_test_subj  = np.array(y_true_test_subj)
y_pred_test_subj  = np.array(y_pred_test_subj)
y_proba_test_subj = np.array(y_proba_test_subj)


acc_subj  = accuracy_score(y_true_test_subj, y_pred_test_subj)
f1_subj   = f1_score(y_true_test_subj, y_pred_test_subj, average="weighted")
prec_subj = precision_score(y_true_test_subj, y_pred_test_subj, average="weighted", zero_division=0)
rec_subj  = recall_score(y_true_test_subj, y_pred_test_subj, average="weighted")

print(f"\nSubject-level Test Accuracy:  {acc_subj  * 100:.2f}%")
print(f"Subject-level Test F1:        {f1_subj   * 100:.2f}%")
print(f"Subject-level Test Precision: {prec_subj * 100:.2f}%")
print(f"Subject-level Test Recall:    {rec_subj  * 100:.2f}%")
print(f"Number of test subjects:      {len(y_true_test_subj)}")


cm_subj = confusion_matrix(y_true_test_subj, y_pred_test_subj)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_subj, annot=True, fmt="d", cmap="Blues",
            xticklabels=["AD", "FTD", "HC"],
            yticklabels=["AD", "FTD", "HC"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Test Subject-level (Combined OvR)")
plt.tight_layout()
plt.show()


y_true_test_subj_bin = label_binarize(y_true_test_subj, classes=[0, 1, 2])

plt.figure(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_test_subj_bin[:, i], y_proba_test_subj[:, i])
    auc_score   = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names_roc[i]}  AUC = {auc_score:.3f}")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Subject-level (Combined OvR)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


