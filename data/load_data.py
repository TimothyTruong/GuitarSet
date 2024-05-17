from datasets import load_dataset
import librosa
import os
import numpy as np
import tensorflow as tf

def extract_data(batch):
    # Map the local file path, corresponding to the dataset's audio file path, along wiht the key (label)
    
    # Paths to audio files and keys
    audio_paths = [os.path.join(batch['location'][i], 'audio.wav') for i in range(len(batch['location']))]
    print(audio_paths)
    keys = batch['key']
    return {'audio': audio_paths, 'key': keys}


dataset = load_dataset('amaai-lab/MusicBench')
dataset = dataset.map(extract_data, batched=True)

# Load the dataset
print(dataset['train'][0])