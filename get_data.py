import requests
from tqdm import tqdm
import zipfile
import os
from pathlib import Path

def download_zip(url, destination):
    '''
    Downloads the zip file specified to the destination
    '''

    # Get the file object itself
    response = requests.get(url, stream=True)
    # Get total size of file
    total_bytes = int(response.headers.get('content-length', 0))

    # Open file, create tqdm instance for stream downloading
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_bytes,
        unit='B',
        unit_scale=True,
        unit_divisor=1024) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))

def extract_zip(path):
    '''
    Extracts and deletes zip file
    '''

    with zipfile.ZipFile(path, 'r') as zip_obj:
        file_list = zip_obj.namelist()

        # Create the progress bar
        with tqdm(total=len(file_list), desc=f'Extracting {path}') as pbar:
            for file in file_list:
                # Extract each file
                zip_obj.extract(file, "data")
                pbar.update(1)

    os.remove(path)

if __name__ == "__main__":
    #### MAKE DATA PATH (if doesn't exist)
    Path("/data").mkdir(exist_ok=True)

    #### METADATA
    print("Downloading metadata...")
    file_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    destination_path = "fma_metadata.zip"
    download_zip(file_url, destination_path)

    print(f"Downloaded to {destination_path}!")

    print(f"Extracting metadata...")
    extract_zip(destination_path)

    #### MUSIC DATA
    print("Downloading music data...")
    file_url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    destination_path = "fma_medium.zip"
    download_zip(file_url, destination_path)

    print(f"Downloaded to {destination_path}!")

    print(f"Extracting metadata...")
    extract_zip(destination_path)

    #### SMALLER TEST MUSIC DATA (uncomment if you want to use)
    # print("Downloading music data...")
    # file_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    # destination_path = "fma_small.zip"
    # download_zip(file_url, destination_path)

    # print(f"Downloaded to {destination_path}!")

    # print(f"Extracting metadata...")
    # extract_zip(destination_path)