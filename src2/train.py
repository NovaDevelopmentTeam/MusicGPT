# nova_music2/src/train.py
import os
import gc
import sys
import torch
torch.backends.cudnn.benchmark = True  # Performance optimization
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AudioGPT
from dataset import AudioDataset, collate_fn
from config import Config
# HuggingFace-Tokenizer global import
from transformers import AutoTokenizer

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Wrapper for dataset
def tokenize(text: str):
    # Return list of token IDs
    return tokenizer.encode(text, add_special_tokens=True)


def compute_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error Loss between model output and target spectrogram."""
    return F.mse_loss(output, target)


def save_checkpoint(model: torch.nn.Module, optimizer: optim.Optimizer, epoch: int, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: Config,
    genre_name: str
):
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        steps = len(dataloader)
        print(f"🌀 [Genre: {genre_name}] Epoche {epoch}/{config.epochs} — Schritte: {steps}")

        for step, (mel_spec, text_tokens, filenames) in enumerate(dataloader, start=1):
            if mel_spec is None:
                continue

            mel_spec = mel_spec.to(device, dtype=torch.float)
            text_tokens = text_tokens.to(device)

            optimizer.zero_grad()
            output = model(mel_spec, text_tokens)
            loss = compute_loss(output, mel_spec)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"   [{step}/{steps}] {filenames[0]} | Loss: {loss.item():.4f}")

            # Free GPU memory
            del mel_spec, text_tokens, output, loss
            gc.collect()

        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"📉 Durchschnittlicher Verlust Epoche {epoch}: {avg_loss:.4f}")

        # Checkpoint speichern
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            ckpt_name = f"checkpoint{epoch}_state.pt" if epoch < config.epochs else "final_state.pt"
            save_path = os.path.join("models", genre_name, ckpt_name)
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"💾 Modell gespeichert: {save_path}")


def train_all_genres(data_root: str = "data"):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    for genre in sorted(os.listdir(data_root)):
        genre_path = os.path.join(data_root, genre)
        if not os.path.isdir(genre_path):
            continue

        # Audio- und Text-Dateien sammeln
        audio_files = [os.path.join(genre_path, f)
                       for f in os.listdir(genre_path)
                       if f.lower().endswith((".wav", ".mp3"))]
        descriptions = []
        for audio_path in audio_files:
            txt_path = os.path.splitext(audio_path)[0] + ".txt"
            if os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8") as fh:
                    descriptions.append(fh.read().strip())
            else:
                descriptions.append("")

        if not audio_files:
            print(f"⚠️ Keine Audiodateien in {genre_path} gefunden.")
            continue

        print(f"🔊 === Training Genre: {genre} ===")

        dataset = AudioDataset(
            audio_paths=audio_files,
            text_descriptions=descriptions,
            tokenizer=tokenize,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            augment=config.use_augmentation,
            max_time_steps=config.max_time_steps
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        model = AudioGPT(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        train_model(model, dataloader, optimizer, device, config, genre_name=genre)


if __name__ == "__main__":
    choice = input("Songs scrapen? (Nur wenn 'data' leer) y/n: ").strip().lower()
    if choice in ["y", "yes"]:
        import scrape_songs as scraper
        scraper.scrape_genres(genres=["rock", "blues", "pop", "jazz"])

    train_all_genres(data_root="data")
