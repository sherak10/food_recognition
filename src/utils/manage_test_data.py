from keras.preprocessing import image
import numpy as np


def load_img_as_array(test_data_path, food_name, images_name, target_size):
    images_data = []
    for image_name in images_name:
        # skip hidden files
        if not image_name.startswith('.'):
            img = image.load_img(
                test_data_path + food_name + '/' + image_name,
                target_size=target_size)
            img_arr = image.img_to_array(img)
            img_arr = np.expand_dims(img_arr, axis=0)
            images_data.append(img_arr)

    # pass the list of multiple images np.vstack()
    return np.vstack(images_data)


# image classification accuracy
def calculcate_accuracy(predicted_classes, i):
    num_valid_prediction = 0
    num_test_samples = len(predicted_classes)

    for predicted_class in predicted_classes:
        if predicted_class[i]:
            num_valid_prediction += 1

    return round(num_valid_prediction / num_test_samples, 3)
