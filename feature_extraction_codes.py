import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, stft


#%% 4) Custom functions

def extract_features(signal, fs=500):
    """
    Extract time-domain, frequency-domain, and time-frequency features from a 1D EEG signal.

    Parameters:
    - signal: 1D array-like
    - fs: int, sampling frequency in Hz (default: 500 for this dataset)

    Returns:
    - features: 1D numpy array of length 59
    """
    N = len(signal)

    # --- Time-domain ---
    T1 = np.max(signal)
    T2 = np.min(signal)
    T3 = np.mean(signal)
    T4 = np.var(signal)
    T5 = np.std(signal)
    T6 = np.mean(np.abs(signal - T3))
    T7 = np.sqrt(np.mean(signal**2))
    T8 = np.mean(np.abs(np.diff(signal)))
    T9 = np.sum(signal**2)
    T10 = np.ptp(signal)
    T11 = np.sum(np.abs(np.diff(signal)))
    hist, _ = np.histogram(signal, bins='auto', density=True)
    probabilities = hist / np.sum(hist)
    T12 = entropy(probabilities, base=2)
    T13 = np.trapezoid(signal)
    T14 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    T15 = np.sum(np.arange(N) * signal**2) / np.sum(signal**2)
    peaks, _ = find_peaks(signal)
    T16 = len(peaks)
    T17 = np.sum(np.sqrt(1 + np.diff(signal)**2))
    T18 = np.sum(signal**2) / N
    T19 = np.sum(signal[:-1] * signal[1:] < 0) / N
    T20 = skew(signal)
    T21 = kurtosis(signal)
    T22 = np.sum((signal[:-2] < signal[1:-1]) & (signal[1:-1] > signal[2:]))
    T23 = np.sum((signal[:-2] > signal[1:-1]) & (signal[1:-1] < signal[2:]))
    T24 = np.max(signal) - np.min(signal)
    T25 = np.sum(np.abs(np.diff(signal)))
    T26 = np.sqrt(np.var(np.diff(signal)) / T4)
    T27 = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal))) / T26

    # --- Frequency-domain ---
    fft_val = np.abs(fft(signal))
    freqs = fftfreq(len(signal), d=1/fs)
    freqs = freqs[:len(freqs)//2]
    fft_val = fft_val[:len(freqs)]
    power_vals = fft_val**2
    power_vals = 10 * np.log10(power_vals + 1e-12)

    F1 = np.max(power_vals)
    F2 = freqs[np.argmax(power_vals)]
    cumulative_power = np.cumsum(power_vals)
    total_power = cumulative_power[-1]
    F3 = freqs[np.where(cumulative_power >= total_power / 2)[0][0]]
    F4 = np.sum(freqs * power_vals) / np.sum(power_vals)
    threshold = 0.5 * np.max(power_vals)
    low_cutoff = np.min(freqs[np.where(power_vals >= threshold)])
    high_cutoff = np.max(freqs[np.where(power_vals >= threshold)])
    F5 = high_cutoff - low_cutoff
    F6 = np.sum((power_vals - power_vals)**2)
    epsilon = 1e-10
    norm_power_vals = (power_vals + epsilon) / (np.sum(power_vals) + epsilon)
    F7 = -np.sum(norm_power_vals * np.log2(norm_power_vals))
    F8 = np.sqrt(np.sum((freqs - F4)**2 * power_vals) / np.sum(power_vals))
    F9 = np.sum((freqs - F4)**3 * power_vals) / (F8**3 * np.sum(power_vals))
    F10 = np.sum((freqs - F4)**4 * power_vals) / (F8**4 * np.sum(power_vals))
    F11 = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]
    F12 = freqs[np.where(cumulative_power >= 1.15 * total_power)[0][0]] if np.any(cumulative_power >= 1.15 * total_power) else None
    F13 = freqs[np.argmax(power_vals)]
    F14 = np.sum(power_vals[(freqs >= 0.6) & (freqs <= 2.5)]) / np.sum(power_vals)
    H_high = np.max(power_vals[freqs >= 0.5 * fs / 2])
    H_low = np.min(power_vals[freqs >= 0.5 * fs / 2])
    F15 = H_high / H_low
    F16 = 1 - np.sum(freqs * power_vals * power_vals) / (
        np.sqrt(np.sum(freqs * power_vals) * np.sum(freqs * power_vals))
    )
    delta_power = np.sum(power_vals[(freqs >= 0.5) & (freqs <= 4)])
    theta_power = np.sum(power_vals[(freqs > 4) & (freqs <= 8)])
    alpha_power = np.sum(power_vals[(freqs > 8) & (freqs <= 13)])
    beta_power  = np.sum(power_vals[(freqs > 13) & (freqs <= 30)])
    total_power = np.sum(power_vals)
    F17 = delta_power / total_power
    F18 = theta_power / total_power
    F19 = alpha_power / total_power
    F20 = beta_power  / total_power

    # --- Time-frequency (DWT) ---
    W1 = pywt.wavedec(signal, 'db4', level=5)
    W2 = [np.sum(np.square(c)) for c in W1]
    TF1 = np.sum(W2)
    W3 = np.sum(W2)
    TF2 = entropy(np.array(W2) / W3, base=2)
    TF3 = [W2[i] / W2[i + 1] for i in range(len(W2) - 1)]
    TF4 = [np.var(d) for d in W1]

    features = np.array([
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17,
        T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F13, F14, F15, F16,
        F17, F18, F19, F20,
        TF1, TF2,
        TF3[0], TF3[1], TF3[2], TF3[3], TF3[4],
        TF4[0], TF4[1], TF4[2], TF4[3], TF4[4], TF4[5]
    ])

    return features


FEATURE_NAMES = (
    [f"T{i}" for i in range(1, 28)] +
    ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10",
     "F11","F13","F14","F15","F16","F17","F18","F19","F20"] +
    ["TF1","TF2",
     "TF3_0","TF3_1","TF3_2","TF3_3","TF3_4",
     "TF4_0","TF4_1","TF4_2","TF4_3","TF4_4","TF4_5"]
)
