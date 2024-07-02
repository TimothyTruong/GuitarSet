# from datasets import load_dataset
# import librosa
# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import json
# from sklearn.preprocessing import LabelEncoder

# # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# data_path = r"E:\Projects\MusicBench_Data\extracted_files\datashare"
# save_dir = r"\data\spectrograms"
# metadata_file = r"\data\metadata.json"

# def load_dataset():
#     # Load the dataset
#     dataset = load_dataset('jamendo', 'audio', split='train[:10%]')
    
#     # Extract the audio file paths and keys
#     dataset = dataset.map(extract_data)
    
#     return dataset

# def extract_data(batch):
#     # Map the local file path, corresponding to the dataset's audio file path, along wiht the key (label)
    
#     # Paths to audio files and keys
#     audio_paths = [os.path.normpath(os.path.join(data_path, loc)) for loc in batch['location']]
#     keys = [' '.join(key) if isinstance(key, list) else key for key in batch['key']]
    
#     return {'audio_path': audio_paths, 'key': keys}


# def convert_to_mel_spectrogram(audio_path):
#     audio, sr = librosa.load(audio_path, sr=16000)
#     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
#     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     return mel_spectrogram_db

# def visualize_spectrogram(spectrogram, sr=16000, hop_length=512):
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel spectrogram')
#     plt.tight_layout()
#     plt.show()

# # Convert .wav to Mel spectrogram and save
# def convert_to_mel_spectrogram(audio_path, sr=16000, n_mels=128, fmax=8000):
#     audio, _ = librosa.load(audio_path, sr=sr)
#     mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax)
#     mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     return mel_spectrogram_db

# def preprocess_and_save(data_path, save_dir, metadata_file):
#     os.makedirs(save_dir, exist_ok=True)
#     metadata = []
    
#     for file_name in os.listdir(data_path):
#         if file_name.endswith('.wav'):
#             audio_path = os.path.join(data_path, file_name)
#             spectrogram = convert_to_mel_spectrogram(audio_path)
            
#             # Save the spectrogram as .npy file
#             spectrogram_file = os.path.join(save_dir, file_name.replace('.wav', '.npy'))
#             np.save(spectrogram_file, spectrogram)
            
#             # Extract the key label from file name or other logic
#             key_label = extract_key_label(file_name)  # Define this function based on your naming convention
            
#             metadata.append({
#                 'spectrogram_path': spectrogram_file,
#                 'label': key_label
#             })
    
#     # Save metadata to a JSON file
#     with open(metadata_file, 'w') as f:
#         json.dump(metadata, f)

# def extract_key_label(file_name):
#     # Implement your logic to extract the key label from the file name
#     return file_name.split('_')[0] + " " + file_name.split('_')[1]

# # Run the preprocessing
# preprocess_and_save(data_path, save_dir, metadata_file)

