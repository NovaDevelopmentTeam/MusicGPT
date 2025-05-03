import os
import subprocess
import csv
import json

def download_song(genre, url, index):
    genre_folder = f"data/{genre}"
    os.makedirs(genre_folder, exist_ok=True)
    filename = f"{genre_folder}/song{index}.wav"

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
        # Search for the genre on YouTube
        search_query = f"{genre} music"
        search_url = f"https://www.youtube.com/results?search_query={search_query}"

        # Dynamically scrape the first few results
        try:
            result = subprocess.run(
                ["yt-dlp", "-j", "--flat-playlist", search_url],
                capture_output=True, text=True, check=True
            )

            # Parse the JSON output and extract the URLs
            video_data = result.stdout.splitlines()
            links = []
            
            for line in video_data:
                try:
                    # Try to parse JSON and extract the URL if available
                    data = json.loads(line)
                    if 'url' in data:
                        links.append(data['url'])
                except json.JSONDecodeError:
                    # Skip lines that can't be parsed as JSON
                    continue

            # Ensure we have unique links by using a set
            links = list(set(links))

            # Download songs based on the links (limiting to 5 songs per genre)
            for idx, link in enumerate(links[:5]):  # Limit to 5 songs per genre
                path = download_song(genre, link, idx)
                if path:
                    all_files.append([path])

        except subprocess.CalledProcessError:
            print(f"Error scraping genre: {genre}")

    # Save to CSV
    with open("song_files.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file_paths"])
        writer.writerows(all_files)

if __name__ == "__main__":
    # Example genres
    genres = ["rock", "blues", "pop", "jazz"]

    scrape_genres(genres)
