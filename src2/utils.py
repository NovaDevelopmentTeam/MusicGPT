# nova_music2/src/utils.py
import librosa
import numpy as np
import torch
import os

def load_audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=256):
    try:
        # Überprüfe, ob audio_path ein Array ist
        if isinstance(audio_path, np.ndarray):
            y = audio_path
        else:
            # Falls es sich um einen Dateipfad handelt, versuchen wir, das Audio zu laden
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Die Datei {audio_path} wurde nicht gefunden.")
            y, sr = librosa.load(audio_path, sr=sr)

        # Mel-Spektrogramm berechnen
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    except Exception as e:
        print(f"Fehler beim Laden oder Verarbeiten der Audiodatei {audio_path}: {e}")
        return None  # Im Fehlerfall None zurückgeben

def get_masked_with_pad_tensor(seq_len, query_tensor, key_tensor, pad_token=0):
    padding_mask = (query_tensor != pad_token).float()
    look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len)).float()
    return padding_mask, look_ahead_mask
