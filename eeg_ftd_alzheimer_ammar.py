#%% 0) Project description
"""
Classification of subjects with Alzheimer's disease,
Frontotemporal dementia and Healthy controls by EEG signals.
06/2026

Approach  : Hand-crafted features (time-domain + frequency-domain + DWT wavelet)
            + inter-channel coherence features
            Classical ML : SVM, Random Forest, XGBoost  (subject-wise majority vote)
            Deep Learning: EEGNet, ShallowConvNet, SpectrogramCNN

Split     : Subject-wise stratified 70% train / 10% val / 20% test (no data leakage)
            All epochs from one subject stay in the same split.

Dataset link: https://doi.org/10.18112/openneuro.ds004504.v1.0.8
"""


#%% 1) Imports

from pathlib import Path
from collections import Counter
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pywt
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch, coherence, resample
from scipy.signal import spectrogram as scipy_spectrogram
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


#%% 2) Config and paths

BASE_DIR     = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASET_DIR  = BASE_DIR / "data" / "ds004504"
OUTPUT_DIR   = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

FS          = 500          # original sampling frequency (Hz)
FS_DL       = 128          # downsample target for DL models
EPOCH_SEC   = 20           # epoch length in seconds
EPOCH_SAMP  = FS * EPOCH_SEC          # 10 000 samples per epoch
N_TIME_DL   = FS_DL * EPOCH_SEC      # 2 560 samples after downsampling
N_CH        = 19
N_CLASSES   = 3

RANDOM_STATE = 42
EPOCHS_DL    = 100
BATCH_SIZE   = 16
LR           = 1e-4

CLASS_NAMES = ["Alzheimer", "FTD", "Control"]
CH_NAMES    = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
               "T3","C3","Cz","C4","T4",
               "T5","P3","Pz","P4","T6","O1","O2"]
GROUP_MAP   = {"A": 0, "F": 1, "C": 2}

COH_BANDS   = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
               "alpha": (8.0, 13.0), "beta":  (13.0, 30.0)}
CH_PAIRS    = list(combinations(range(N_CH), 2))   # 171 pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


#%% 3) Data loading and segmentation

def load_eeg(set_path):
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    available = [ch for ch in CH_NAMES if ch in raw.ch_names]
    raw.pick_channels(available, ordered=True)
    data = raw.get_data() * 1e6    # V → µV
    return data

def segment_epochs(data):
    n_ch, n_samp = data.shape
    n_ep = n_samp // EPOCH_SAMP
    data = data[:, :n_ep * EPOCH_SAMP]
    return data.reshape(n_ch, n_ep, EPOCH_SAMP).transpose(1, 0, 2)   # (ep, ch, t)

participants = pd.read_csv(DATASET_DIR / "participants.tsv", sep="\t")

raw_epochs, labels, subject_ids = [], [], []

for _, row in participants.iterrows():
    sub_id = row["participant_id"]
    group  = row["Group"].strip().upper()[0]
    path   = DATASET_DIR / "derivatives" / sub_id / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"

    if not path.exists():
        print(f"  [skip] {sub_id}")
        continue

    print(f"  Loading {sub_id} ({CLASS_NAMES[GROUP_MAP[group]]}) ...", end=" ", flush=True)
    try:
        data = load_eeg(path)
    except Exception as e:
        print(f"ERROR: {e}"); continue

    epochs = segment_epochs(data)
    print(f"{epochs.shape[0]} epochs")

    for ep in epochs:
        raw_epochs.append(ep.astype(np.float32))
        labels.append(GROUP_MAP[group])
        subject_ids.append(sub_id)

X_raw    = np.stack(raw_epochs)              # (N, 19, 10000)
y        = np.array(labels)
subjects = np.array(subject_ids)

print(f"\nLoaded: {X_raw.shape}  |  AD={np.sum(y==0)}, FTD={np.sum(y==1)}, CN={np.sum(y==2)}")
print(f"Unique subjects: {len(set(subjects))}")


#%% 4) Feature extraction — per-channel hand-crafted + inter-channel coherence

def extract_features(signal, fs=FS):
    """59 features per channel: 27 time-domain + 20 frequency-domain + 12 time-frequency (DWT)."""
    N = len(signal)

    # Time-domain
    T1  = np.max(signal);   T2 = np.min(signal);  T3 = np.mean(signal)
    T4  = np.var(signal);   T5 = np.std(signal)
    T6  = np.mean(np.abs(signal - T3))
    T7  = np.sqrt(np.mean(signal**2))
    T8  = np.mean(np.abs(np.diff(signal)))
    T9  = np.sum(signal**2)
    T10 = np.ptp(signal)
    T11 = np.sum(np.abs(np.diff(signal)))
    hist, _ = np.histogram(signal, bins='auto', density=True)
    T12 = entropy(hist / hist.sum(), base=2)
    T13 = np.trapezoid(signal)
    T14 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    T15 = np.sum(np.arange(N) * signal**2) / (np.sum(signal**2) + 1e-12)
    T16 = len(find_peaks(signal)[0])
    T17 = np.sum(np.sqrt(1 + np.diff(signal)**2))
    T18 = np.sum(signal**2) / N
    T19 = np.sum(signal[:-1] * signal[1:] < 0) / N
    T20 = skew(signal);  T21 = kurtosis(signal)
    T22 = np.sum((signal[:-2] < signal[1:-1]) & (signal[1:-1] > signal[2:]))
    T23 = np.sum((signal[:-2] > signal[1:-1]) & (signal[1:-1] < signal[2:]))
    T24 = np.max(signal) - np.min(signal)
    T25 = np.sum(np.abs(np.diff(signal)))
    T26 = np.sqrt(np.var(np.diff(signal)) / (T4 + 1e-12))
    T27 = np.sqrt(np.var(np.diff(np.diff(signal))) / (np.var(np.diff(signal)) + 1e-12)) / (T26 + 1e-12)

    # Frequency-domain
    fft_val = np.abs(fft(signal))
    freqs   = fftfreq(N, d=1/fs)[:N//2]
    fft_val = fft_val[:len(freqs)]
    pwr     = 10 * np.log10(fft_val**2 + 1e-12)
    cum_pwr = np.cumsum(pwr)
    tot     = cum_pwr[-1]

    F1  = np.max(pwr)
    F2  = freqs[np.argmax(pwr)]
    F3  = freqs[np.where(cum_pwr >= tot / 2)[0][0]]
    F4  = np.sum(freqs * pwr) / (np.sum(pwr) + 1e-12)
    thr = 0.5 * F1
    F5  = np.max(freqs[pwr >= thr]) - np.min(freqs[pwr >= thr])
    F6  = 0.0
    norm_p = (pwr + 1e-10) / (np.sum(pwr) + 1e-10)
    F7  = -np.sum(norm_p * np.log2(norm_p))
    F8  = np.sqrt(np.sum((freqs - F4)**2 * pwr) / (np.sum(pwr) + 1e-12))
    F9  = np.sum((freqs - F4)**3 * pwr) / ((F8**3 + 1e-12) * (np.sum(pwr) + 1e-12))
    F10 = np.sum((freqs - F4)**4 * pwr) / ((F8**4 + 1e-12) * (np.sum(pwr) + 1e-12))
    F11 = freqs[np.where(cum_pwr >= 0.95 * tot)[0][0]]
    F13 = freqs[np.argmax(pwr)]
    F14 = np.sum(pwr[(freqs >= 0.6) & (freqs <= 2.5)]) / (np.sum(pwr) + 1e-12)
    h   = pwr[freqs >= 0.5 * fs / 2]
    F15 = np.max(h) / (np.min(h) + 1e-12)
    F16 = 1 - np.sum(freqs * pwr**2) / (np.sqrt(np.sum(freqs * pwr) * np.sum(freqs * pwr)) + 1e-12)
    F17 = np.sum(pwr[(freqs >= 0.5) & (freqs <= 4)])   / (np.sum(pwr) + 1e-12)
    F18 = np.sum(pwr[(freqs > 4)   & (freqs <= 8)])    / (np.sum(pwr) + 1e-12)
    F19 = np.sum(pwr[(freqs > 8)   & (freqs <= 13)])   / (np.sum(pwr) + 1e-12)
    F20 = np.sum(pwr[(freqs > 13)  & (freqs <= 30)])   / (np.sum(pwr) + 1e-12)

    # Time-frequency (DWT, db4 level-5)
    coeffs  = pywt.wavedec(signal, 'db4', level=5)
    energies = [np.sum(c**2) for c in coeffs]
    tot_e   = sum(energies) + 1e-12
    TF1 = tot_e
    TF2 = entropy(np.array(energies) / tot_e, base=2)
    TF3 = [energies[i] / (energies[i+1] + 1e-12) for i in range(len(energies)-1)]
    TF4 = [np.var(c) for c in coeffs]

    return np.array([
        T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,
        T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,
        F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F13,F14,F15,F16,
        F17,F18,F19,F20,
        TF1,TF2,
        TF3[0],TF3[1],TF3[2],TF3[3],TF3[4],
        TF4[0],TF4[1],TF4[2],TF4[3],TF4[4],TF4[5],
    ])


def extract_coherence(epoch_data):
    """Mean coherence per channel pair per band → 171 × 4 = 684 features."""
    feats = []
    for i, j in CH_PAIRS:
        f, Cxy = coherence(epoch_data[i], epoch_data[j], fs=FS, nperseg=FS*2)
        for flo, fhi in COH_BANDS.values():
            mask = (f >= flo) & (f <= fhi)
            feats.append(float(np.mean(Cxy[mask])) if mask.any() else 0.0)
    return np.array(feats)


print("Extracting features...")
all_feats = []
for idx, epoch in enumerate(X_raw):
    if idx % 500 == 0:
        print(f"  epoch {idx}/{len(X_raw)}")
    per_ch = np.concatenate([extract_features(epoch[c]) for c in range(N_CH)])   # 1 121
    coh    = extract_coherence(epoch)                                              #   684
    all_feats.append(np.concatenate([per_ch, coh]))

X = np.nan_to_num(np.array(all_feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
print(f"Feature matrix: {X.shape}  (1 121 per-channel + 684 coherence = 1 805 total)")


#%% 5) Subject-wise train / val / test split  (70 / 10 / 20)

def subject_split(subjects, y_arr):
    unique_subs = np.array(sorted(set(subjects)))
    sub_labels  = np.array([y_arr[subjects == s][0] for s in unique_subs])

    subs_train, subs_temp, _, _ = train_test_split(
        unique_subs, sub_labels, test_size=0.30,
        stratify=sub_labels, random_state=RANDOM_STATE)
    labels_temp = np.array([y_arr[subjects == s][0] for s in subs_temp])

    subs_val, subs_test, _, _ = train_test_split(
        subs_temp, labels_temp, test_size=2/3,
        stratify=labels_temp, random_state=RANDOM_STATE)

    return set(subs_train), set(subs_val), set(subs_test)


def epoch_mask(subjects, sub_set):
    return np.array([s in sub_set for s in subjects])


def majority_vote(y_true, y_pred, y_prob, subjects):
    sv_true, sv_pred, sv_prob = [], [], []
    for sub in sorted(set(subjects)):
        mask = subjects == sub
        sv_true.append(y_true[mask][0])
        sv_pred.append(Counter(y_pred[mask]).most_common(1)[0][0])
        sv_prob.append(y_prob[mask].mean(axis=0))
    return np.array(sv_true), np.array(sv_pred), np.array(sv_prob)


train_subs, val_subs, test_subs = subject_split(subjects, y)
print(f"Split → Train: {len(train_subs)} | Val: {len(val_subs)} | Test: {len(test_subs)} subjects")


#%% 6) Classical ML — SVM, Random Forest, XGBoost

tr_mask  = epoch_mask(subjects, train_subs)
val_mask = epoch_mask(subjects, val_subs)
te_mask  = epoch_mask(subjects, test_subs)

X_tr,  y_tr,  s_tr  = X[tr_mask],  y[tr_mask],  subjects[tr_mask]
X_val, y_val, s_val = X[val_mask], y[val_mask], subjects[val_mask]
X_te,  y_te,  s_te  = X[te_mask],  y[te_mask],  subjects[te_mask]

scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_val  = scaler.transform(X_val)
X_te   = scaler.transform(X_te)

pca   = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_tr  = pca.fit_transform(X_tr)
X_val = pca.transform(X_val)
X_te  = pca.transform(X_te)
print(f"PCA: {X_tr.shape[1]} components retained (95% variance)")

ml_models = {
    "SVM": SVC(kernel="rbf", C=10, gamma="scale",
               class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", n_jobs=-1, random_state=RANDOM_STATE),
}

ml_results = {}
for name, model in ml_models.items():
    print(f"\n── {name} ──")
    model.fit(X_tr, y_tr)

    te_pred = model.predict(X_te)
    te_prob = model.predict_proba(X_te)
    ep_acc  = accuracy_score(y_te, te_pred)
    ep_f1   = f1_score(y_te, te_pred, average="macro", zero_division=0)
    ep_auc  = roc_auc_score(label_binarize(y_te, classes=[0,1,2]),
                            te_prob, multi_class="ovr", average="macro")

    sv_true, sv_pred, sv_prob = majority_vote(y_te, te_pred, te_prob, s_te)
    sv_acc = accuracy_score(sv_true, sv_pred)
    sv_f1  = f1_score(sv_true, sv_pred, average="macro", zero_division=0)
    sv_auc = roc_auc_score(label_binarize(sv_true, classes=[0,1,2]),
                           sv_prob, multi_class="ovr", average="macro")

    print(f"  Epoch  → Acc={ep_acc:.3f}  F1={ep_f1:.3f}  AUC={ep_auc:.3f}")
    print(f"  Subject→ Acc={sv_acc:.3f}  F1={sv_f1:.3f}  AUC={sv_auc:.3f}")
    print(classification_report(sv_true, sv_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(sv_true, sv_pred, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name} (Subject-level)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cm_{name.replace(' ','_').lower()}.png", dpi=150)
    plt.close()

    ml_results[name] = dict(ep_acc=ep_acc, ep_f1=ep_f1, ep_auc=ep_auc,
                            sv_acc=sv_acc, sv_f1=sv_f1, sv_auc=sv_auc,
                            sv_true=sv_true, sv_pred=sv_pred)


#%% 7) Deep Learning models — EEGNet, ShallowConvNet, SpectrogramCNN

class EEGNet(nn.Module):
    """EEGNet: Lawhern et al. (2018). Input: (batch, 1, C, T)."""
    def __init__(self, n_classes=N_CLASSES, n_ch=N_CH, n_time=N_TIME_DL,
                 F1=8, D=2, F2=16, kern_length=250, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            flat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_time))).view(1,-1).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = self.block2(self.block1(x))
        return self.classifier(x.view(x.size(0), -1))


class ShallowConvNet(nn.Module):
    """ShallowConvNet: Schirrmeister et al. (2017). Input: (batch, 1, C, T)."""
    def __init__(self, n_classes=N_CLASSES, n_ch=N_CH, n_time=N_TIME_DL,
                 n_filters=40, filter_time=25, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, filter_time), bias=False)
        self.spatial  = nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False)
        self.bn       = nn.BatchNorm2d(n_filters)
        self.pool     = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop     = nn.Dropout(dropout)
        with torch.no_grad():
            flat = self._features(torch.zeros(1, 1, n_ch, n_time)).view(1,-1).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def _features(self, x):
        x = self.drop(self.pool(torch.log(torch.clamp(self.bn(self.spatial(self.temporal(x)))**2, 1e-6))))
        return x

    def forward(self, x):
        return self.classifier(self._features(x).view(x.size(0), -1))


class SpectrogramCNN(nn.Module):
    """2-D CNN on per-channel log-power spectrograms. Input: (batch, C, freq, time)."""
    def __init__(self, n_classes=N_CLASSES, n_ch=N_CH, freq_bins=65, time_frames=78):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),   nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def compute_spectrograms(X, fs=FS_DL, nperseg=256, noverlap=192):
    """Returns (N, C, freq_bins, time_frames) log-power spectrograms."""
    f, t, _ = scipy_spectrogram(X[0, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    out = np.zeros((len(X), N_CH, len(f), len(t)), dtype=np.float32)
    for i in range(len(X)):
        for c in range(N_CH):
            _, _, Sxx = scipy_spectrogram(X[i, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
            out[i, c] = np.log10(Sxx + 1e-10)
    return out, len(f), len(t)


def get_class_weights(y_arr):
    classes, counts = np.unique(y_arr, return_counts=True)
    w = 1.0 / counts; w = w / w.sum() * len(classes)
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


def train_model(model_fn, X_tensor, model_name):
    tr_idx  = np.where(epoch_mask(subjects, train_subs))[0]
    val_idx = np.where(epoch_mask(subjects, val_subs))[0]
    te_idx  = np.where(epoch_mask(subjects, test_subs))[0]

    X_tr  = X_tensor[tr_idx];  y_tr_t  = torch.tensor(y[tr_idx],  dtype=torch.long)
    X_val = X_tensor[val_idx]; y_val_t = torch.tensor(y[val_idx], dtype=torch.long)
    X_te  = X_tensor[te_idx];  y_te_t  = torch.tensor(y[te_idx],  dtype=torch.long)

    # Normalise on train statistics
    mean = X_tr.mean(); std = X_tr.std() + 1e-8
    X_tr  = (X_tr  - mean) / std
    X_val = (X_val - mean) / std
    X_te  = (X_te  - mean) / std

    tr_loader  = DataLoader(TensorDataset(X_tr,  y_tr_t),  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    te_loader  = DataLoader(TensorDataset(X_te,  y_te_t),  batch_size=BATCH_SIZE, shuffle=False)

    model     = model_fn().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(y[tr_idx]))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_DL)

    best_val_loss, best_weights = float("inf"), None

    for ep in range(1, EPOCHS_DL + 1):
        model.train()
        tr_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); tr_loss += loss.item()
        tr_loss /= len(tr_loader)
        scheduler.step()

        # Validation — monitoring only, no early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{EPOCHS_DL} | Train={tr_loss:.4f} | Val={val_loss:.4f} | Best={best_val_loss:.4f}")

    model.load_state_dict(best_weights)
    model.eval()

    all_pred, all_prob = [], []
    with torch.no_grad():
        for xb, _ in te_loader:
            probs = F.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
            all_pred.extend(np.argmax(probs, axis=1))
            all_prob.extend(probs)

    y_pred = np.array(all_pred); y_prob = np.array(all_prob)
    y_true = y[te_idx]; s_te   = subjects[te_idx]

    ep_acc = accuracy_score(y_true, y_pred)
    ep_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ep_auc = roc_auc_score(label_binarize(y_true, classes=[0,1,2]),
                           y_prob, multi_class="ovr", average="macro")

    sv_true, sv_pred, sv_prob = majority_vote(y_true, y_pred, y_prob, s_te)
    sv_acc = accuracy_score(sv_true, sv_pred)
    sv_f1  = f1_score(sv_true, sv_pred, average="macro", zero_division=0)
    sv_auc = roc_auc_score(label_binarize(sv_true, classes=[0,1,2]),
                           sv_prob, multi_class="ovr", average="macro")

    print(f"  [Test] Epoch  → Acc={ep_acc:.3f}  F1={ep_f1:.3f}  AUC={ep_auc:.3f}")
    print(f"  [Test] Subject→ Acc={sv_acc:.3f}  F1={sv_f1:.3f}  AUC={sv_auc:.3f}")

    return dict(ep_acc=ep_acc, ep_f1=ep_f1, ep_auc=ep_auc,
                sv_acc=sv_acc, sv_f1=sv_f1, sv_auc=sv_auc,
                sv_true=sv_true, sv_pred=sv_pred)


#%% 8) Deep Learning training

# Downsample 500 Hz → 128 Hz
print("Downsampling raw epochs for DL models...")
X_dl = resample(X_raw, N_TIME_DL, axis=2).astype(np.float32)
print(f"  X_dl: {X_dl.shape}")

# EEGNet  — input (batch, 1, 19, 2560)
print("\n══ EEGNet ══")
X_eeg = torch.tensor(X_dl[:, np.newaxis, :, :], dtype=torch.float32)
dl_results = {}
dl_results["EEGNet"] = train_model(lambda: EEGNet(), X_eeg, "EEGNet")
print(classification_report(dl_results["EEGNet"]["sv_true"],
                            dl_results["EEGNet"]["sv_pred"],
                            target_names=CLASS_NAMES, zero_division=0))

# ShallowConvNet
print("\n══ ShallowConvNet ══")
dl_results["ShallowConvNet"] = train_model(lambda: ShallowConvNet(), X_eeg, "ShallowConvNet")
print(classification_report(dl_results["ShallowConvNet"]["sv_true"],
                            dl_results["ShallowConvNet"]["sv_pred"],
                            target_names=CLASS_NAMES, zero_division=0))

# SpectrogramCNN  — input (batch, 19, freq_bins, time_frames)
print("\n══ SpectrogramCNN ══")
print("  Computing spectrograms...")
X_spec, freq_bins, time_frames = compute_spectrograms(X_dl)
print(f"  Spectrogram shape: {X_spec.shape}")
X_spec_t = torch.tensor(X_spec, dtype=torch.float32)
dl_results["SpectrogramCNN"] = train_model(
    lambda: SpectrogramCNN(freq_bins=freq_bins, time_frames=time_frames),
    X_spec_t, "SpectrogramCNN")
print(classification_report(dl_results["SpectrogramCNN"]["sv_true"],
                            dl_results["SpectrogramCNN"]["sv_pred"],
                            target_names=CLASS_NAMES, zero_division=0))


#%% 9) EDA — Band-power topography maps (delta / theta / alpha / beta)

from mne.viz import plot_topomap

TOPO_BANDS = {
    "Delta (0.5–4 Hz)":  (0.5,  4.0),
    "Theta (4–8 Hz)":    (4.0,  8.0),
    "Alpha (8–13 Hz)":   (8.0, 13.0),
    "Beta (13–30 Hz)":  (13.0, 30.0),
}
GROUP_LABELS = {"A": "Alzheimer's Disease", "F": "Frontotemporal Dementia", "C": "Healthy Control"}

def subject_band_powers(data_dir):
    """Load each subject's EEG and compute mean Welch PSD per band per channel."""
    import glob
    group_powers = {"A": [], "F": [], "C": []}
    mne_info = None

    for _, row in participants.iterrows():
        sub_id = row["participant_id"]
        group  = row["Group"].strip().upper()[0]
        path   = (data_dir / "derivatives" / sub_id / "eeg"
                  / f"{sub_id}_task-eyesclosed_eeg.set")
        if not path.exists():
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
            avail = [ch for ch in CH_NAMES if ch in raw.ch_names]
            raw.pick_channels(avail, ordered=True)
            raw.set_channel_types({ch: "eeg" for ch in avail})
            raw.set_montage("standard_1020", on_missing="warn")
        except Exception as e:
            print(f"  [skip] {sub_id}: {e}"); continue

        if mne_info is None:
            mne_info = raw.info

        sub_bp = []
        for flo, fhi in TOPO_BANDS.values():
            psds, _ = raw.compute_psd(method="welch", fmin=flo, fmax=fhi,
                                      n_fft=2048, verbose=False).get_data(return_freqs=True)
            sub_bp.append(np.mean(psds * 1e12, axis=1))   # µV²/Hz → mean per channel
        group_powers[group].append(np.stack(sub_bp, axis=1))   # (n_ch, n_bands)

    for g in group_powers:
        group_powers[g] = np.stack(group_powers[g], axis=0) if group_powers[g] else None
    return group_powers, mne_info

print("Computing band-power topomaps...")
gp, mne_info = subject_band_powers(DATASET_DIR)

band_names = list(TOPO_BANDS.keys())
for b_idx, band_name in enumerate(band_names):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Mean Band Power Topography — {band_name}", fontsize=14, fontweight="bold")

    all_data   = np.concatenate([gp[g][:, :, b_idx] for g in ["A","F","C"] if gp[g] is not None])
    grand_mean = all_data.mean()
    norm_vals  = [gp[g][:, :, b_idx].mean(axis=0) / grand_mean for g in ["A","F","C"] if gp[g] is not None]
    max_dev    = max(abs(v - 1.0).max() for v in norm_vals)
    vmin, vmax = 1.0 - max_dev, 1.0 + max_dev

    for ax, g in zip(axes, ["A","F","C"]):
        mean_power = gp[g][:, :, b_idx].mean(axis=0) / grand_mean
        im, _ = plot_topomap(mean_power, mne_info, axes=ax, vlim=(vmin, vmax),
                             cmap="RdBu_r", show=False)
        ax.set_title(f"{GROUP_LABELS[g]}\n(n={gp[g].shape[0]})", fontsize=10)

    plt.colorbar(im, ax=axes.tolist(), shrink=0.6, label="Relative Power (grand-mean normalized)")
    out = OUTPUT_DIR / f"topomap_{band_name.split()[0].lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")

print("Topomaps done.\n")


#%% 10) Final results and comparison plots

# Save confusion matrices
for name, res in {**ml_results, **dl_results}.items():
    cm = confusion_matrix(res["sv_true"], res["sv_pred"], labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name} (Subject-level)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cm_{name.replace(' ','_').lower()}_subject.png", dpi=150)
    plt.close()

# Summary table
rows = []
for name, res in {**ml_results, **dl_results}.items():
    rows.append({"Model": name,
                 "Ep Acc":  round(res["ep_acc"],  4),
                 "Ep F1":   round(res["ep_f1"],   4),
                 "Ep AUC":  round(res["ep_auc"],  4),
                 "Sub Acc": round(res["sv_acc"],  4),
                 "Sub F1":  round(res["sv_f1"],   4),
                 "Sub AUC": round(res["sv_auc"],  4)})

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "all_results_summary.csv", index=False)

print("\n══ Final Summary ══")
print(summary.to_string(index=False))

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
for ax, (cols, title) in zip(axes, [
    (["Ep Acc","Ep F1","Ep AUC"],   "Epoch-level (Test Set)"),
    (["Sub Acc","Sub F1","Sub AUC"],"Subject-level / Majority Vote (Test Set)"),
]):
    x = np.arange(3); w = 0.13
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.bar(x + i*w, [row[c] for c in cols], w,
               label=row["Model"], color=colors[i], alpha=0.85)
    ax.set_xticks(x + w*2.5)
    ax.set_xticklabels(["Accuracy","F1 (Macro)","AUC-ROC"], fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.suptitle("EEG Dementia Classification — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "all_models_comparison.png", dpi=150)
plt.close()

print(f"\nAll figures saved to: {OUTPUT_DIR}")
