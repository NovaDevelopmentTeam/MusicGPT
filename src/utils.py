import librosa
import numpy as np
import torch

def load_audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=256): # sr=22050, n_mels=128
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def get_masked_with_pad_tensor(seq_len, query_tensor, key_tensor, pad_token=0):
    padding_mask = (query_tensor != pad_token).float()
    look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len)).float()
    return padding_mask, look_ahead_mask
