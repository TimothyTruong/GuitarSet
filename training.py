from preprocessing.load_dataset import make_dataset
from models.model import build_key_model
dataset = make_dataset('train')

# #split into training and validation set
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset = dataset.take(train_size)
# val_dataset = dataset.skip(train_size)

dataset_size = 0
for _ in dataset:
    dataset_size += 1

print(dataset_size)


# model = build_key_model()