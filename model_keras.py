########## Fruits Data set ##############
import os
from keras import models,layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

path=r'C:\Users\Anu\Downloads\datasets\fruits-360' #Thats my Training and Validation Data Path

train_dir=os.path.join(path,'training')
validation_dir=os.path.join(path,'validation')

###### model Creation
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(65,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',
              metrics=['acc'])


image_datagen=ImageDataGenerator(rescale=1./255,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rotation_range=40,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

train_datagen=image_datagen.flow_from_directory(train_dir,
                            target_size=(100,100),
                            batch_size=20,
                            class_mode='categorical',
                            shuffle=False)
validation_datagen=image_datagen.flow_from_directory(validation_dir,
                                 target_size=(100,100),
                                 batch_size=20,
                                 class_mode='categorical',
                                 shuffle=False)
history=model.fit_generator(
        train_datagen,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_datagen,
        validation_steps=50)
model.save('Fruits_model')

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.show() 