from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
loaded_data = np.load('validation_data.npz')
x_train = loaded_data['X']
y_train = loaded_data['Y']
x_train = x_train.reshape(-1, 150, 150, 1)
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,
        zca_whitening=False,  
        rotation_range = 30,  
        zoom_range = 0.2, 
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip = True,  
        vertical_flip=False)  

datagen.fit(x_train)
np.savez('validation_data.1.npz', X=x_train, Y=y_train)