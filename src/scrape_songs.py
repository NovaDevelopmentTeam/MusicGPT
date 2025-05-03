import os
import subprocess
import csv
import json
import signal
import sys
import glob

# Speicher aller begonnenen Downloads
partial_files = []

def download_song(genre, url, index):
    genre_folder = f"data/{genre}"
    os.makedirs(genre_folder, exist_ok=True)
    filename = f"{genre_folder}/song{index}.wav"

    print(f"Downloading {genre} song {index}...")
    try:
        subprocess.run([
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "-o", f"{genre_folder}/song{index}.%(ext)s",
            url
        ], check=True)
        return filename
    except subprocess.CalledProcessError:
        print(f"Download failed for: {url}")
        return None

def cleanup_partial_files():
    print("\nAbbruch erkannt. Unvollständige Dateien werden gelöscht...")
    for filepath in partial_files:
        base = os.path.splitext(filepath)[0]
        for f in glob.glob(base + "*"):
            try:
                os.remove(f)
                print(f"Gelöscht: {f}")
            except Exception as e:
                print(f"Fehler beim Löschen von {f}: {e}")
    print("Aufräumen abgeschlossen.")

def scrape_genres(genres):
    all_files = []

    try:
        for genre in genres:
            search_query = f"{genre} music"
            search_url = f"https://www.youtube.com/results?search_query={search_query}"

            print(f"Suche nach Genre: {genre}...")
            result = subprocess.run(
                ["yt-dlp", "-j", "--flat-playlist", search_url],
                capture_output=True, text=True, check=True
            )

            video_data = result.stdout.splitlines()
            links = []

            for line in video_data:
                try:
                    data = json.loads(line)
                    if 'url' in data:
                        links.append(data['url'])
                except json.JSONDecodeError:
                    continue

            links = list(set(links))

            for idx, link in enumerate(links[:5]):
                genre_folder = f"data/{genre}"
                partial_filename = f"{genre_folder}/song{idx}"
                partial_files.append(partial_filename)
                path = download_song(genre, link, idx)
                if path:
                    all_files.append([path])

    except KeyboardInterrupt:
        cleanup_partial_files()
        sys.exit(1)

    with open("song_files.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file_paths"])
        writer.writerows(all_files)

if __name__ == "__main__":
    genres = ["rock", "blues", "pop", "jazz"]
    scrape_genres(genres)
