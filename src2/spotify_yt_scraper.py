import os
import subprocess
import csv
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='YOUR_SPOTIPY_CLIENT_ID', client_secret='YOUR_SPOTIPY_CLIENT_SECRET'))

def get_spotify_songs_by_genre(genre, limit=5):
    # Suche nach Songs im angegebenen Genre
    results = sp.search(q=f"genre:{genre}", type='track', limit=limit)
    songs = []
    for track in results['tracks']['items']:
        song_name = track['name']
        song_artist = track['artists'][0]['name']
        songs.append(f"{song_name} {song_artist}")
    return songs

def download_song(genre, url, index):
    genre_folder = f"data/{genre}"
    os.makedirs(genre_folder, exist_ok=True)
    filename = f"{genre_folder}/song{index}.mp3"

    print(f"Downloading {genre} song {index}...")

    try:
        subprocess.run([
            "yt-dlp",
            "-v",  # Für detaillierte Ausgabe
            "-x", "--audio-format", "mp3",
            "-o", f"{genre_folder}/song{index}.%(ext)s",
            url
        ], check=True)
        return filename
    except subprocess.CalledProcessError:
        print(f"Download failed for: {url}")
        return None

def scrape_genres(genres):
    all_files = []

    for genre in genres:
        # Hole Spotify-Songs für das Genre
        print(f"Getting songs for genre: {genre}")
        spotify_songs = get_spotify_songs_by_genre(genre)

        for song in spotify_songs:
            search_query = song  # Der Songname von Spotify
            search_url = f"https://www.youtube.com/results?search_query={search_query}"

            # Dynamisch die ersten YouTube-Ergebnisse durchsuchen
            try:
                result = subprocess.run(
                    ["yt-dlp", "-j", "--flat-playlist", search_url],
                    capture_output=True, text=True, check=True
                )

                # JSON-Daten der YouTube-Ergebnisse parsen
                video_data = result.stdout.splitlines()
                links = []

                for line in video_data:
                    try:
                        # JSON-Daten extrahieren und URL speichern
                        data = json.loads(line)
                        if 'url' in data:
                            links.append(data['url'])
                    except json.JSONDecodeError:
                        continue

                # Stelle sicher, dass die Links einzigartig sind
                links = list(set(links))

                # Lade die Songs herunter (beschränke auf 5 Songs pro Genre)
                for idx, link in enumerate(links[:5]):  # Beschränkung auf 5 Videos pro Genre
                    path = download_song(genre, link, idx)
                    if path:
                        all_files.append([path])

            except subprocess.CalledProcessError:
                print(f"Error scraping YouTube for genre: {genre}")

    # Speichern der Ergebnisse in einer CSV-Datei
    with open("song_files.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file_paths"])
        writer.writerows(all_files)

if __name__ == "__main__":
    # Beispiel-Genres
    genres = ["rock", "blues", "pop", "jazz"]
    scrape_genres(genres)
