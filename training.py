from preprocessing.load_dataset import make_dataset
from models.model import build_key_model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

BATCH_SIZE = 32
NUM_EPOCHS = 10


dataset = make_dataset('train', batch_size=BATCH_SIZE)

# dataset_size = 0
# for _ in dataset:
#     dataset_size += 1

# dataset_size = dataset_size * BATCH_SIZE

# #split into training and validation set
# train_size = int(0.8 * dataset_size)
# val_size = dataset_size- train_size

# train_dataset = dataset.take(train_size)
# val_dataset = dataset.skip(train_size)

# first = dataset.take(1)
# print(first)
model = build_key_model()

# # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
terminate_on_nan = TerminateOnNaN()

model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[terminate_on_nan])
# # model.fit(train_dataset, validation_data=val_dataset, epochs=NUM_EPOCHS, callbacks=[early_stopping, terminate_on_nan])
# model.save("keyai_model.keras")