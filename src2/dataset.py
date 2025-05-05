# nova_music2/src/dataset.py

import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import load_audio_to_mel_spectrogram

def default_augment(audio, sr):
    # Hier echte Augmentations einfügen: Pitch, Time-Stretch, Rauschen...
    return audio

class AudioDataset(Dataset):
    def __init__(self, audio_paths, text_descriptions, tokenizer, sr=22050, n_mels=256, augment=False, max_time_steps=512):
        self.audio_paths = audio_paths
        self.texts = text_descriptions
        self.tokenizer = tokenizer
        self.sr = sr
        self.n_mels = n_mels
        self.augment = augment
        self.max_time_steps = max_time_steps

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        text = self.texts[idx]
        try:
            mel = load_audio_to_mel_spectrogram(path, sr=self.sr, n_mels=self.n_mels)
            if mel is None:
                raise RuntimeError("Spektrogramm konnte nicht berechnet werden.")
            if self.augment:
                mel = default_augment(mel, self.sr)
            if mel.shape[1] > self.max_time_steps:
                mel = mel[:, :self.max_time_steps]
            mel = mel.T  # [T, n_mels]
            tokens = torch.tensor(self.tokenizer(text), dtype=torch.long)
            return mel, tokens, os.path.basename(path)
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {path}: {e}")
            return None, None, None

# Collate-Funktion für DataLoader

def collate_fn(batch):
    specs, texts, names = zip(*batch)
    filtered = [(s, t, n) for s, t, n in batch if s is not None]
    if not filtered:
        return None, None, None
    specs, texts, names = zip(*filtered)

    # Jedes Spektrogramm in Tensor konvertieren (falls noch numpy)
    specs = [torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s for s in specs]

    # Spektrogramme paddden: [B, T, n_mels]
    specs = pad_sequence(specs, batch_first=True)

    # Text-Token-Sequenzen paddden: [B, T_text]
    texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return specs, texts, names
