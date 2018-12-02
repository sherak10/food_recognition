import os
from tensorflow import ConfigProto, Session
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import categorical_crossentropy
from keras.optimizers import Adadelta
from food_recognition.src.utils.vgg16_model import create_model
import matplotlib.pyplot as plt

# use if you are running on a PC with many GPU-s
# needs to be at the beginning of the file
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# the GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# prevents GPU overhead
config = ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = .4
session = Session(config=config)

train_images_path = '../../data/processed/train'
model_path = '../../models/model.h5'

# define image input shape and number of classes
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
num_classes = 18

# dimensionality
units = 256

# define batch size, epochs and steps per epoch to use in training
batch_size = 5
epochs = 10
# number of images / batch size (approximately)
steps_per_epoch = 2200

my_model = create_model(input_shape, num_classes, units)

# weights and layers from VGG part will be hidden in the summary
# they will be fit during the training
# my_model.summary()

# generate data from images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
        train_images_path,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle='True')

# train data
my_model.compile(
        loss=categorical_crossentropy,
        optimizer=Adadelta(),
        metrics=['accuracy'])

history = my_model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# serialize weights to HDF5
my_model.save_weights(model_path)
print("Saved model to disk")
