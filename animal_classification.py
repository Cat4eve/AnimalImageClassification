import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Setting the number of classes and image size
num_classes = 10
image_size = (224, 224)

# Loading the MobileNet model (pre-trained on ImageNet)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adding custom layers on top of MobileNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Creating the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freezing initial layers if desired
for layer in base_model.layers:
    layer.trainable = False

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Setting the directory for the dataset
dataset_dir = 'animals'

# Setting the batch size and number of epochs
batch_size = 32
num_epochs = 10

# Loading the training and validation datasets
train_generator = train_datagen.flow_from_directory(
    dataset_dir + '/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    dataset_dir + '/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Saving the trained model
model.save('animal_classification_model.h5')

# Saving the list of class names
class_names = list(train_generator.class_indices.keys())
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
