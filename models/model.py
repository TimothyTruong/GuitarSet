import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, LSTM, Dropout, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal

def build_key_model():
    model = Sequential()

    initializer = HeNormal()
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)

    # The input into the model will be mel spectrograms of shape (None, None, 1) - to allow for variable length of audio

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(None, None, 1), kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print(model.output_shape)

    print(model.output_shape)

    # Reshape the output for LSTM
    # Assume the output shape here from the last pooling layer is (batch_size, time_steps, features_per_step)
    model.add(Reshape((-1, 128)))

    model.add(LSTM(128, return_sequences=True, kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(LSTM(128, kernel_initializer=initializer))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(24, activation='softmax', kernel_initializer=initializer))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model 

def print_summmary(s):
    with open("summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: print(x, file=f))

if __name__ == "__main__":
    model = build_key_model()
    model.summary(print_fn=print_summmary)
    model.summary()


