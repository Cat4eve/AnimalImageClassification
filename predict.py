import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Load and preprocess the input image
img_path = 'animals/validation/squirrel/OIP-26sJIg9yACYfTUaeZdAlbwHaFj.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
preprocessed_img = preprocess_input(img_array)
preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

# Perform prediction
predictions = model.predict(preprocessed_img)
# Get the predicted class
predicted_class = np.argmax(predictions)

# Load the list of class names
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Get the actual class name from the image path
actual_class_name = img_path.split('/')[-2]

# Visualize the image
image = plt.imread(img_path)

# Plot the image with predicted and actual class names
plt.imshow(image)
plt.axis('off')
plt.text(10, 30, f'Predicted class: {class_names[predicted_class]}',
         fontsize=12,
         color='white',
         bbox=dict(facecolor='black', alpha=0.5))
plt.text(10, 60, f'Actual class: {actual_class_name}',
         fontsize=12,
         color='white',
         bbox=dict(facecolor='black', alpha=0.5))
plt.show()
