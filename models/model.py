import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM, Dropout, Reshape
from tensorflow.keras.models import Sequential

def build_key_model():
    model = Sequential()

    # The input into the model will be mel spectrograms of shape (None, None, 1) - to allow for variable length of audio

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(None, None, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(model.output_shape)

    print(model.output_shape)

    # Reshape the output for LSTM
    # Assume the output shape here from the last pooling layer is (batch_size, time_steps, features_per_step)
    model.add(Reshape((-1, 128)))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    
    # Given an input of a song, the model will find the key of the song
    model.add(Dense(24, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model 

def print_summmary(s):
    with open("summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: print(x, file=f))

if __name__ == "__main__":
    model = build_key_model()
    model.summary(print_fn=print_summmary)
    model.summary()


