import os
import requests
from tqdm import tqdm

# URL of the Google Speech Commands dataset
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Local path to save the dataset
save_path = "speech_commands_v0.02.tar.gz"
extract_path = "speech_commands"


# Function to download the dataset
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
    else:
        print(f"Downloaded {save_path}")


# Function to extract the dataset
def extract_tar_gz(file_path, extract_path):
    import tarfile
    if file_path.endswith("tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted to {extract_path}")


# Download the dataset
if not os.path.exists(save_path):
    download_file(url, save_path)

# Extract the dataset
if not os.path.exists(extract_path):
    extract_tar_gz(save_path, extract_path)

print("Dataset is ready!")
