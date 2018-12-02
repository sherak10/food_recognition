from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model


def create_model(input_shape, num_classes, units):
    # load pretrained vgg16 model
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # model_vgg16_conv.summary()

    # use the generated model for your input
    input = Input(shape=input_shape, name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(units, activation='relu', name='fc1')(x)
    x = Dense(units, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # create your own model
    return Model(input=input, output=x)
