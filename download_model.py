import os
import requests
from tqdm import tqdm

def download_file(url, local_path):
    """Downloads a file from a URL to a local path with a progress bar."""
    # Ensure the target directory exists
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
        return

    print(f"Downloading from {url} to {local_path}...")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 8192

            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=os.path.basename(local_path)) as progress_bar:
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download.")
                os.remove(local_path) # Clean up
            else:
                print(f"Successfully downloaded {local_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

if __name__ == "__main__":
    MODEL_URL = "https://ai4code.blob.core.windows.net/repohyper/model_10.pt"
    MODEL_PATH = os.path.join("models", "model_10.pt")
    download_file(MODEL_URL, MODEL_PATH)