# nova_music2/train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AudioGPT
from dataset import AudioDataset
import librosa
import numpy as np
import os
import gc
from config import Config

def load_audio_to_mel_spectrogram(audio_path, sr=22050, n_mels=256):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def compute_loss(output, target):
    return F.mse_loss(output, target)

def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

def train_model(model, dataloader, optimizer, device, epochs=50, genre_name="???", start_epoch=0, model_dir="models"):
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        print(f"\n🌀 [Genre: {genre_name}] → Starte Epoche {epoch + 1}/{epochs} mit {len(dataloader)} Samples")

        for i, (mel_spec, file_name) in enumerate(dataloader):
            mel_spec = mel_spec.to(device)
            mel_spec = torch.squeeze(mel_spec, 1)
            optimizer.zero_grad()
            output = model(mel_spec)
            loss = compute_loss(output, mel_spec)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"   [{i + 1}/{len(dataloader)}] Datei: {file_name[0]} | Loss: {loss.item():.4f}")

            # Speicherbereinigung
            del mel_spec, output, loss
            gc.collect()

        avg_loss = total_loss / len(dataloader)
        print(f"📉 [Epoche {epoch + 1}] Durchschnittlicher Verlust: {avg_loss:.4f}")

        # Checkpoint speichern (alle 5 Epochen)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_name = f"checkpoint{epoch + 1}_state.pt" if (epoch + 1) < epochs else "final_state.pt"
            save_path = os.path.join(model_dir, checkpoint_name)
            save_checkpoint(model, optimizer, epoch + 1, save_path)
            print(f"💾 Modell gespeichert unter: {save_path}")

def train_all_genres(data_root="data", epochs=50):
    genres = [genre for genre in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, genre))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Konfiguration laden ===
    config = Config()

    for genre in genres:
        print(f"\n🔊 === Starte Training für Genre: {genre} ===")
        genre_path = os.path.join(data_root, genre)
        audio_files = [os.path.join(genre_path, f) for f in os.listdir(genre_path) if f.endswith((".wav", ".mp3"))]

        if not audio_files:
            print(f"⚠️ Keine Audiodateien in {genre_path} gefunden.")
            continue

        print(f"📂 Gefundene Dateien für '{genre}': {len(audio_files)}")

        dataset = AudioDataset(audio_files)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        model = AudioGPT(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # === Speicherort vorbereiten ===
        model_dir = os.path.join("models", genre)
        os.makedirs(model_dir, exist_ok=True)

        # === Modell wiederherstellen, falls vorhanden ===
        latest_checkpoint = None
        start_epoch = 0

        checkpoints = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("checkpoint") and f.endswith("_state.pt")
        ], key=lambda x: int(x.replace("checkpoint", "").replace("_state.pt", "")))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(model_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"♻️ Fortsetze ab Checkpoint: {latest_checkpoint} (Epoche {start_epoch + 1})")
        else:
            print(f"🆕 Kein Checkpoint für '{genre}' gefunden. Starte neues Training.")

        # === Training starten ===
        train_model(model, dataloader, optimizer, device, epochs,
                    genre_name=genre, start_epoch=start_epoch, model_dir=model_dir)

if __name__ == "__main__":
    allow_scrape = input("Do you want to scrape songs first? (Only use if dataset 'Data' is empty!) y/n: ").lower().strip()
    if allow_scrape in answerY:
        import scrape_songs as scraper
        scraper.scrape_genres(genres=["rock", "blues", "pop", "jazz"])
    elif allow_scrape in answerN:
        train_all_genres(data_root="data", epochs=50)
    else:
        print("Wrong Input. Please enter a valid answer.")
