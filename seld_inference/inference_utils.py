"""
inference_utils.py

Utility functions for SELD inference.
Extracted from the original utils.py for standalone use.
"""

import librosa
import librosa.feature
import numpy as np
import torch


def load_audio(audio_file, sampling_rate):
    """
    Loads an audio file.
    Args:
        audio_file (str): Path to the audio file.
        sampling_rate (int): Target sampling rate
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(path=audio_file, sr=sampling_rate, mono=False)
    return audio_data, sr


def compute_msiv(audio_data, sr: int = 24000, n_fft: int = 512, win_length: int = 512,
                 hop_length: int = 300, nb_mels: int = 64, fmin: int = 50):
    """
    Extract Mid-Side Intensity Vector. 

    Args:
        audio_data (np.ndarray): Stereo audio signal with shape (2, n_samples).
        sr (int): Sampling rate. Default is 24000.
        n_fft (int): Number of FFT points for STFT. Default is 512.
        hop_length (int): Hop length between successive STFT frames. Default is 300.
        win_length (int): Window length for STFT. Default is 512.
        nb_mels (int): Number of Mel bands to generate. Default is 64.
        fmin (int): Minimum frequency (in Hz). Default is 50Hz.

    Returns:
        np.ndarray: (time_bins, freq_bins)
    """
    eps = 1e-8

    mid = (audio_data[0] + audio_data[1]) / 2
    side = (audio_data[0] - audio_data[1]) / 2
    M_stft = librosa.stft(mid, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True)
    S_stft = librosa.stft(side, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True)

    I_num = np.real(M_stft * np.conj(S_stft))
    E = eps + np.abs(M_stft)**2 + np.abs(S_stft)**2
    I_norm = I_num / E

    if nb_mels:
        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=fmin)
        I_norm = mel_fb.dot(I_norm)

    return I_norm.T


def compute_binaural_msc_recursive(stfts, lambda_: float = 0.8, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64):
    """
    Compute Magnitude-Squared Coherence (MSC) spectrogram using time-recursive averaging.
    """
    if not (0 < lambda_ < 1):
        raise ValueError(f"Parameter lambda_ must be in the interval (0, 1), but got lambda_={lambda_}.")

    epsilon = 1e-8
    X = stfts
    T, F = X.shape[1], X.shape[2]

    S_11 = np.zeros((F,), dtype=np.complex64)
    S_22 = np.zeros((F,), dtype=np.complex64)
    S_12 = np.zeros((F,), dtype=np.complex64)
    gamma = np.zeros((T, F), dtype=np.float32)

    for t in range(T):
        X_l = X[0, t, :]
        X_r = X[1, t, :]

        S_11 = lambda_ * S_11 + (1 - lambda_) * (X_l * np.conjugate(X_l))
        S_22 = lambda_ * S_22 + (1 - lambda_) * (X_r * np.conjugate(X_r))
        S_12 = lambda_ * S_12 + (1 - lambda_) * (X_l * np.conjugate(X_r))

        numerator = np.abs(S_12)**2
        denominator = (S_11 * S_22).real + epsilon
        gamma[t, :] = numerator / denominator

    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50)
        gamma = gamma.dot(mel_filter_bank.T)

    return gamma


def compute_salsa_slite(stfts, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64, fmax: int = 2000):
    """
    Compute the stereo version of SALSA-Lite (Normalized IPD).
    """
    X = stfts.T

    c = 343  # speed of sound
    delta = 2 * np.pi * sr / (n_fft * c)
    n_bins = n_fft // 2 + 1
    freq_vector = np.arange(n_bins)
    freq_vector[0] = 1
    freq_vector = freq_vector[:, None, None]

    phase_vector = np.angle(X[:, :, 1:] * np.conj(X[:, :, 0, None]))
    phase_vector = phase_vector / (delta * freq_vector)
    phase_vector = phase_vector.T

    if fmax is not None:
        upper_bin = int(np.floor(fmax * n_fft / float(sr)))
        phase_vector[:, :, upper_bin:] = 0

    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50)
        phase_vector = np.dot(phase_vector, mel_filter_bank.T)

    return phase_vector


def compute_ipd_features(stfts, sr: int = 24000, n_fft: int = 512, nb_mels: int = 64, max_freq: int = 2000):
    """
    Compute the interchannel phase differences (IPDs) via sine and cosine.
    """
    stft_left = stfts[0].copy()
    stft_right = stfts[1].copy()

    if max_freq is not None:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask = freqs > max_freq
        stft_left[:, mask] = 0
        stft_right[:, mask] = 0

    if nb_mels:
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=nb_mels, fmin=50)
        stft_left = np.dot(stft_left, mel_filter_bank.T)
        stft_right = np.dot(stft_right, mel_filter_bank.T)

    ipd = np.angle(stft_left) - np.angle(stft_right)
    ipd_sin = np.sin(ipd)
    ipd_cos = np.cos(ipd)

    return np.stack((ipd_sin, ipd_cos), axis=0)


def extract_stereo_features(audio, sr: int = 24000, n_fft: int = 512, hop_length: int = 300, win_length: int = 512, 
                            nb_mels: int = 64, max_freq: int = 2000, use_gamma: bool = False, use_ipd: bool = False, 
                            use_iv: bool = False, use_slite: bool = False, use_ms: bool = False):
    """
    Extract stereo audio features. Mel spectrograms are always extracted.
    Optional features: gamma (MSC), ipd, iv (Mid-Side IV), slite (SALSA-Lite), ms (Mid-Side spectrogram).

    Args:
        audio (np.ndarray): Stereo audio signal with shape (2, n_samples).
        sr (int): Sampling rate. Default is 24000.
        n_fft (int): Number of FFT points. Default is 512.
        hop_length (int): Hop length. Default is 300.
        win_length (int): Window length. Default is 512.
        nb_mels (int): Number of Mel bands. Default is 64.
        max_freq (int): Frequency threshold for IPD. Default is 2000.
        use_gamma (bool): Compute MSC feature.
        use_ipd (bool): Compute IPD features.
        use_iv (bool): Compute Mid-Side IVs.
        use_slite (bool): Compute SALSA-Lite.
        use_ms (bool): Compute Mid-Side spectrograms.

    Returns:
        np.ndarray: (n_feature_channels, time_bins, freq_bins)
    """
    if audio.shape[0] != 2:
        raise ValueError("Input audio must have two channels (shape: (2, n_samples)).")

    # Compute STFT
    stfts = [
        librosa.stft(y=np.asfortranarray(channel), n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann', pad_mode='reflect')
        for channel in audio
    ]
    stfts = np.stack(stfts, axis=0)
    stfts = np.transpose(stfts, (0, 2, 1))

    # Compute mel spectrograms
    power_spec = np.abs(stfts) ** 2
    mel_specs = []
    for channel_power in power_spec:
        pspec = channel_power.T
        if nb_mels:
            mel_spec = librosa.feature.melspectrogram(S=pspec, sr=sr, n_mels=nb_mels, fmin=50)
            log_mel = librosa.power_to_db(mel_spec)
            log_mel = log_mel.T
        else:
            log_mel = librosa.power_to_db(pspec)
            log_mel = log_mel.T
        mel_specs.append(log_mel)
    mel_specs = np.stack(mel_specs, axis=0)

    # Mid-Side spectrograms
    if use_ms:
        left, right = audio[0], audio[1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        ms_stft = [librosa.stft(y=np.asfortranarray(mid), n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window='hann', pad_mode='reflect'),
                   librosa.stft(y=np.asfortranarray(side), n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window='hann', pad_mode='reflect')]
        ms_stft = np.stack(ms_stft, axis=0)
        ms_stft = np.transpose(ms_stft, (0, 2, 1))
        ms_power = np.abs(ms_stft) ** 2
        ms_specs = []
        for channel_power in ms_power:
            pspec = channel_power.T
            if nb_mels:
                mel_spec = librosa.feature.melspectrogram(S=pspec, sr=sr, n_mels=nb_mels, fmin=50)
                log_mel = librosa.power_to_db(mel_spec)
                log_mel = log_mel.T
            else:
                log_mel = librosa.power_to_db(pspec)
                log_mel = log_mel.T
            ms_specs.append(log_mel)
        ms_specs = np.stack(ms_specs, axis=0)
        mel_specs = np.concatenate((mel_specs, ms_specs), axis=0)

    # Mid-Side IVs
    if use_iv:
        i_norm = compute_msiv(audio_data=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, nb_mels=nb_mels)
        mel_specs = np.concatenate((mel_specs, np.expand_dims(i_norm, axis=0)), axis=0)

    # Coherence (gamma)
    if use_gamma:
        msc = compute_binaural_msc_recursive(stfts=stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels)
        msc = np.expand_dims(msc, axis=0)
        mel_specs = np.concatenate((mel_specs, msc), axis=0)

    # IPD features
    if use_ipd:
        ipds = compute_ipd_features(stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels, max_freq=max_freq)
        mel_specs = np.concatenate((mel_specs, ipds), axis=0)

    # SALSA-Lite
    if use_slite:
        salsalite = compute_salsa_slite(stfts, sr=sr, n_fft=n_fft, nb_mels=nb_mels, fmax=max_freq)
        mel_specs = np.concatenate((mel_specs, salsalite), axis=0)

    return mel_specs


def get_multiaccdoa_labels(logits, nb_classes, modality=None, dnorm=False, params=None):
    """
    Parse multi-ACCDOA model output into detection results.
    """
    # First event slot
    x0, y0 = logits[:, :, :1*nb_classes], logits[:, :, 1*nb_classes:2*nb_classes]
    x0 = torch.clamp(x0, 0, 1)
    y0 = torch.clamp(y0, -1, 1)
    sed0 = torch.sqrt(x0**2 + y0**2) > 0.5

    dist0 = logits[:, :, 2*nb_classes:3*nb_classes]
    if dnorm and params:
        dist0 = (dist0 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist0 = torch.clamp(dist0, min=0)

    doa0 = logits[:, :, :2*nb_classes]
    dummy_src_id0 = torch.zeros_like(dist0)
    on_screen0 = torch.zeros_like(dist0)

    # Second event slot
    x1, y1 = logits[:, :, 3*nb_classes:4*nb_classes], logits[:, :, 4*nb_classes:5*nb_classes]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, -1, 1)
    sed1 = torch.sqrt(x1**2 + y1**2) > 0.5

    dist1 = logits[:, :, 5*nb_classes:6*nb_classes]
    if dnorm and params:
        dist1 = (dist1 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist1 = torch.clamp(dist1, min=0)

    doa1 = logits[:, :, 3*nb_classes:5*nb_classes]
    dummy_src_id1 = torch.zeros_like(dist1)
    on_screen1 = torch.zeros_like(dist1)

    # Third event slot
    x2, y2 = logits[:, :, 6*nb_classes:7*nb_classes], logits[:, :, 7*nb_classes:8*nb_classes]
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, -1, 1)
    sed2 = torch.sqrt(x2**2 + y2**2) > 0.5

    dist2 = logits[:, :, 8*nb_classes:9*nb_classes]
    if dnorm and params:
        dist2 = (dist2 * params['d_max'] * params['d_std']) + params['d_mean']
    else:
        dist2 = torch.clamp(dist2, min=0)

    doa2 = logits[:, :, 6*nb_classes:8*nb_classes]
    dummy_src_id2 = torch.zeros_like(dist2)
    on_screen2 = torch.zeros_like(dist2)

    return sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2
