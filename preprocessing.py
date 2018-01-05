from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# data augmentation
datagen = ImageDataGenerator(
    rotation_range = 40,      # a value in degrees (0, 180), a range within which to randomly rotate pictures
    width_shift_range = 0.2,  # a fraction range within which to randomly translate pictures horizontally
    height_shift_range = 0.2, # same as above (vertically)
    #rescale = 1.0 / 255,      # a value by which we will multiply the data before any processing
    shear_range = 0.2,        # Shear Intensity (Shear angle in counter-clockwise direction as radians), for shear mapping
    zoom_range = 0.2,         # Range for random zoom, [lower, upper] = [1-zoom_range, 1+zoom_range]
    horizontal_flip = True,   # Randomly flip inputs horizontally
    fill_mode = 'nearest')    # Points outside the boundaries of the input are filled according to the given mode.   "nearest":  aaaaaaaa|abcd|dddddddd


img = load_img('./data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a numpy array with shape (374, 500, 3)
print(x.shape, type(x))

x = x.reshape((1,) + x.shape) # a numpy array with shape (1, 374, 500, 3)
print(x.shape, type(x))


# the .flow( method generate batches of randomly transformed images
# and saves the results to the "./preview/" directory
i = 0
for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir = './preview',
                          save_prefix = 'car',
                          save_format = 'jpeg'):
    i += 1
    if (i > 20):
        break
