import tensorflow as tf
from datasets import load_dataset
import librosa
import os
import numpy as np

# Define the path where the audio files are stored
data_path = r"E:\Projects\MusicBench_Data\extracted_files\datashare"

key_mapping = {
    ('C', 'major'): 1, ('C#', 'major'): 2, ('Db', 'major'): 2, ('D', 'major'): 3, ('D#', 'major'): 4, ('Eb', 'major'): 4,
    ('E', 'major'): 5, ('F', 'major'): 6, ('F#', 'major'): 7, ('Gb', 'major'): 7, ('G', 'major'): 8, ('G#', 'major'): 9,
    ('Ab', 'major'): 9, ('A', 'major'): 10, ('A#', 'major'): 11, ('Bb', 'major'): 11, ('B', 'major'): 12,
    ('C', 'minor'): 13, ('C#', 'minor'): 14, ('Db', 'minor'): 14, ('D', 'minor'): 15, ('D#', 'minor'): 16, ('Eb', 'minor'): 16,
    ('E', 'minor'): 17, ('F', 'minor'): 18, ('F#', 'minor'): 19, ('Gb', 'minor'): 19, ('G', 'minor'): 20, ('G#', 'minor'): 21,
    ('Ab', 'minor'): 21, ('A', 'minor'): 22, ('A#', 'minor'): 23, ('Bb', 'minor'): 23, ('B', 'minor'): 24
}

def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Convert to Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=44100, n_mels=128, fmax=8000)
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def retrieve_data(split='train'):
    data = load_dataset("amaai-lab/MusicBench", split=split)
    
    for loc, key in zip(data['location'], data['key']):
        audio_path = os.path.normpath(os.path.join(data_path, loc))
        key_label = key_mapping[tuple(key)]

        # Check if the file exists to avoid errors
        if os.path.isfile(audio_path):
            spectrogram = preprocess_audio(audio_path)
            # Map musical keys to integers or use them as is if they're already in integer/label form

            yield spectrogram, key_label

def make_dataset(split='train', batch_size=32):
    # Create a generator
    generator = lambda: retrieve_data(split)
    # Define the output types and shapes of the dataset
    output_signature = (
        tf.TensorSpec(shape=(128, None), dtype=tf.float32),  # Adjust the shape based on the actual output of preprocess_audio
        tf.TensorSpec(shape=(), dtype=tf.int32)  # Change dtype if keys are integer labels
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create a Dataset from the generator
    return dataset

if __name__ == "__main__":
    # Create the training dataset
    train_dataset = make_dataset('train')

    # Print one sample from the dataset to check
    print(train_dataset.take(1))
