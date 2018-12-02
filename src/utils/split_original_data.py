from os import listdir, remove
from shutil import copy2
from random import shuffle

data_paths = ['../../data/processed/train/', '../../data/processed/test/']
raw_data_path = '../../data/raw/images/'

for path in data_paths:
    folders = listdir(path)
    folders.sort()

    for folder in folders:
        if not folder.startswith('.'):
            folder_files = listdir(path + folder + "/")
            for filename in folder_files:
                remove(path + folder + "/" + filename)

folders = listdir(path)
folders.sort()

training_volume = 0.8
test_volume = 1 - training_volume

for folder in folders:
    if not folder.startswith('.'):
        folder_files = listdir(raw_data_path + folder + "/")
        # randomize data
        for i in range(100):
            shuffle(folder_files)

        split_index = int(training_volume * len(folder_files))

        training_files = folder_files[:split_index]
        for filename in training_files:
            copy2(
                raw_data_path + folder + "/" + filename, data_paths[0] +
                folder + "/" + filename)

        test_files = folder_files[split_index:]
        for filename in test_files:
            copy2(
                raw_data_path + folder + "/" + filename, data_paths[1] +
                folder + "/" + filename)
