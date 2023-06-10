import tensorflow as tf
import matplotlib as plt

class Animal_Classification_Neural_Network:
    def __init__(self):
        self.model : tf.keras.Sequential = tf.keras.Sequential()


    def __call__(self, X, y=None, training=True):
        if not training: return self.model(X)

        MobileNetV2 : tf.keras.Model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            classes=10
        )

        for layer in MobileNetV2.layers:
            self.model.add(layer)

        return self.__call__(X, y, False)

    def include_images(self):
        self.train_dataset = tf.keras.utils.image_dataset_from_directory('./train', shuffle=True)
        self.test = tf.keras.utils.image_dataset_from_directory('./test', shuffle=True)
