from os import listdir
from keras.backend import categorical_crossentropy
from keras.optimizers import Adadelta
from food_recognition.src.utils.vgg16_model import create_model
from food_recognition.src.utils.manage_test_data import load_img_as_array, \
                                                        calculcate_accuracy
import numpy as np
import matplotlib.pyplot as plt

test_data_path = '../../data/processed/test/'
model_path = '../../models/model.h5'

# define image input shape and number of classes
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
num_classes = 18

# dimensionality
units = 256

# define batch size
batch_size = 5

# create your own model
my_model = create_model(input_shape, num_classes, units)

# load the model we saved
my_model.load_weights(model_path)
my_model.compile(
    loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=['accuracy'])

test_data = {}
test_accuracy = {}
food_names = listdir(test_data_path)
food_names.sort()

# fill test data
for food_name in food_names:
    # skip hidden files
    if not food_name.startswith('.'):
        test_data[food_name] = listdir(test_data_path + food_name + '/')

for i, (food_name, images_name) in enumerate(test_data.items()):
    images_data = load_img_as_array(
        test_data_path,
        food_name,
        images_name,
        (img_rows, img_cols))
    predicted_classes = my_model.predict(
        images_data,
        batch_size=batch_size)
    test_accuracy[food_name] = calculcate_accuracy(
        predicted_classes,
        i)

for i, (food_name, accuracy) in enumerate(test_accuracy.items()):
    print(food_name + ' (' + str(i) + '): ' + str(accuracy))

print('Test accuracy: ' + str(round(np.mean(list(test_accuracy.values())), 3)))

# plot food probability
plt.bar(
    range(num_classes),
    test_accuracy.values(),
    align='center',
    tick_label=range(num_classes))
plt.title('test accuracy')
plt.ylabel('accuracy')
plt.xlabel('food index')
plt.show()
