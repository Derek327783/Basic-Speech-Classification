import json
import os
import librosa


DATASET_PATH = "dataset"
JSON_PATH = "data.json" # Path to store MFCC features
SAMPLES_TO_CONS = 22050 # librosa uses 22050 Hz sample rate by default, so 22050 samples is 1 sec audio

def prepare_dataset(dataset_path,json_path, n_mfcc=13, hop_length = 512, n_fft = 2048):
    data = {
        "mappings" : [], # Word labels
        "labels" : [], # Numerical labels
        "MFCCs" : [], # Feature
        "file_name" : [] # Original file name
    }

    # loop through all dir in dataset
    for i, (dirpath,dirname,filename) in enumerate(os.walk(dataset_path)):
        # dirpath = "dataset/down" etc. filename = list of files in the dirpath
        if i == 5:
            break
        if dirpath is not dataset_path:
            category = dirpath.split("/")[-1]
            data["mappings"].append(category)
            print(f"Processing {category}")
            for audio_files in filename: # Iterate through the files
                file_path = os.path.join(dirpath,audio_files) # Get file path
                file_path = file_path.replace("\\", "/")
                signal,sr = librosa.load(file_path, sr = 22050) # Load audio file
                if len(signal) >= SAMPLES_TO_CONS: # Check audio is at least 1 sec long
                    signal = signal[:SAMPLES_TO_CONS] # Make sure audio is exactly 1 sec long for consistency to fit into NN
                    MFCC = librosa.feature.mfcc(y=signal,n_mfcc=n_mfcc, hop_length = hop_length, n_fft = n_fft)
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCC.T.tolist()) # Need to convert as json cannot store np_array
                    data["file_name"].append(file_path)
                    print(f"Processing {file_path}: {i-1}")

            # Store final data dictionary into json file
    with open(JSON_PATH, "w") as fp:
        json.dump(data,fp,indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH,JSON_PATH)

# data = {
#     "mappings" : [], # Word labels
#     "labels" : [], # Numerical labels
#     "MFCCs" : [], # Feature
#     "file_name" : [] # Original file name
# }
# n_mfcc=13
# hop_length = 512
# n_fft = 2048
#
# for i, (dirpath,dirname,filename) in enumerate(os.walk(DATASET_PATH)):
#     # dirpath = "dataset/down" etc. filename = list of files in the dirpath
#     if i == 2:
#         break
#     if dirpath is not DATASET_PATH:
#         category = dirpath.split("/")[-1]
#         data["mappings"].append(category)
#         print(f"Processing {category}")
#         for audio_files in filename: # Iterate through the files
#             file_path = os.path.join(dirpath,audio_files) # Get file path
#             file_path = file_path.replace("\\","/")
#             try:
#                 signal, sr = librosa.load(file_path, sr=22050)  # Load audio file
#                 if len(signal) >= SAMPLES_TO_CONS:
#                     # Check audio is at least 1 sec long
#                     signal = signal[:SAMPLES_TO_CONS]  # Make sure audio is exactly 1 sec long for consistency to fit into NN
#                     MFCC = librosa.feature.mfcc(y = signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
#                     data["labels"].append(i - 1)
#                     data["MFCCs"].append(MFCC.T.tolist())  # Need to convert as json cannot store np_array
#                     data["file_name"].append(file_path)
#                     print(f"Processing {file_path}: {i - 1}")
#             except:
#                 print("signal error",f"{file_path}")

