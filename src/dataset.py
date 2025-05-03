import torch
from torch.utils.data import Dataset
import os
import librosa
import numpy as np
from utils import load_audio_to_mel_spectrogram  # Stelle sicher, dass diese Funktion korrekt importiert wird

def augment_audio(audio, sr):
    """
    Eine Funktion zur Datenaugmentation von Audiodaten:
    - Pitch shifting
    - Time stretching
    - Rauschen hinzufügen
    - SpecAugment
    """

    # 1. Pitch verschieben (zwischen -2 und 2 Halbtönen)
    audio = librosa.effects.pitch_shift(audio, sr, n_steps=np.random.uniform(-2, 2))

    # 2. Zeitdehnung/Kompression (zwischen 80% und 120% der Originalgeschwindigkeit)
    audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))

    # 3. Rauschen hinzufügen
    noise = np.random.randn(len(audio)) * 0.01
    audio = audio + noise

    # 4. SpecAugment (Maskieren von Zeit- und Frequenzachsen im Mel-Spektrogramm)
    audio = apply_spec_augment(audio, sr)

    return audio

def apply_spec_augment(audio, sr, n_mels=256):
    """
    Wendet SpecAugment an, um Zufalls- und Frequenzmasken auf das Mel-Spektrogramm anzuwenden.
    """
    mel = load_audio_to_mel_spectrogram(audio, sr, n_mels=n_mels)
    mel = mel.numpy()

    # Zeitmaskierung (zufällig ein Teil des Zeitbereichs maskieren)
    time_mask = np.random.randint(0, mel.shape[0] // 2)
    mel[:time_mask, :] = 0

    # Frequenzmaskierung (zufällig ein Teil der Frequenzen maskieren)
    freq_mask = np.random.randint(0, mel.shape[1] // 2)
    mel[:, :freq_mask] = 0

    return torch.tensor(mel, dtype=torch.float32)

def AudioDataset(audio_paths, sr=22050, n_mels=256):
    """
    Ein Dataset, das eine Liste von Audio-Dateipfaden verarbeitet und
    WAV-Dateien in Mel-Spektrogramme umwandelt. Gibt zusätzlich den Dateinamen zurück.
    """ 

    class AudioDatasetClass(Dataset):
        def __init__(self, audio_paths, sr=22050, n_mels=256):
            self.audio_paths = audio_paths
            self.sr = sr
            self.n_mels = n_mels
            self._check_audio_files()

        def _check_audio_files(self):
            missing_files = [path for path in self.audio_paths if not os.path.exists(path)]
            if missing_files:
                raise FileNotFoundError(f"Fehlende Dateien: {', '.join(missing_files)}")

        def __len__(self):
            return len(self.audio_paths)

        def __getitem__(self, idx):
            """
            Gibt das Mel-Spektrogramm und den Dateinamen zurück, wobei Datenaugmentation angewendet wird.
            """
            audio_path = self.audio_paths[idx]
            # Audio laden
            audio, sr = librosa.load(audio_path, sr=self.sr)

            # Datenaugmentation anwenden
            audio = augment_audio(audio, sr)

            # Mel-Spektrogramm berechnen (nach Augmentierung)
            mel_tensor = load_audio_to_mel_spectrogram(audio, sr=self.sr, n_mels=self.n_mels)
            mel_tensor = torch.tensor(mel_tensor.T, dtype=torch.float32)

            # Maximale Zeitstempelbegrenzung (optional für fixe Eingabedimensionen)
            MAX_TIME_STEPS = 2048
            if mel_tensor.shape[0] > MAX_TIME_STEPS:
                mel_tensor = mel_tensor[:MAX_TIME_STEPS, :]

            # Dateiname zurückgeben
            file_name = os.path.basename(audio_path)
            return mel_tensor, file_name  # ⬅️ Dateiname hinzugefügt

    return AudioDatasetClass(audio_paths, sr, n_mels)
