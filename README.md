Animal Classification with MobileNet
This project demonstrates the process of training a deep learning model to classify images of animals using the MobileNet architecture. The model is trained on a dataset of animal images and is capable of predicting the class of a given animal image.

Dataset Architecture
The dataset should be organized as follows:

markdown
Copy code
animals/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
    validation/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
Each class of animals has its own folder containing the corresponding images within the train and validation directories.

Model Architecture
The MobileNet model, pre-trained on the ImageNet dataset, serves as the base model. Custom layers are added on top of the MobileNet architecture to adapt it for animal classification. The model consists of a global average pooling layer, followed by a dense layer with 1024 units and ReLU activation. The final dense layer with softmax activation predicts the class probabilities of the input image.

Training
During training, data augmentation techniques such as rescaling, shearing, zooming, and horizontal flipping are applied to the training images. The images are normalized by dividing the pixel values by 255. The model is trained using the Adam optimizer with categorical cross-entropy loss as the training objective. Training occurs for a specified number of epochs and a given batch size.

Evaluation
The model's performance is evaluated using accuracy as the evaluation metric. The validation dataset is used for evaluation, and the accuracy is reported after each epoch.

Saving the Model
After training, the trained model is saved as "animal_classification_model.h5". Additionally, the list of class names is saved as "class_names.pkl" using the pickle library for later reference.
