import numpy as np
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import os
import cv2
import random
from PIL import Image # for image
import tensorflow as tf
import keras 
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow import keras
from tensorflow.keras import layers, regularizers, models, optimizers
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import array_to_img
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1,NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
from sklearn.metrics import classification_report, confusion_matrix

# Carica il modello salvato
model_path = 'rmi_braintumor_classifier.h5'
model = tf.keras.models.load_model(model_path)

image_data_dir = 'dataset'

# Gather all file paths using glob
train_files = glob.glob(os.path.join(image_data_dir, '*', '*'))

# Settiamo il random seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Shuffle the file list using random.shuffle for better performance
random.shuffle(train_files)

# Extract labels from the file paths (assuming label is the folder name)
train_labels = [os.path.basename(os.path.dirname(file)) for file in train_files]
train_df = pd.DataFrame({
    'image': train_files,
    'label': train_labels
})

# Settiamo il random seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

batch_size = 128
target_size = (224, 224)
validation_split = 0.2
test_split = 0.1

# Load all data
all_data = tf.keras.preprocessing.image_dataset_from_directory(
    image_data_dir,
    image_size=target_size,
    batch_size=batch_size,
    shuffle=True,
    label_mode='categorical'    
)

# Calculate the sizes for splitting the dataset
dataset_size = tf.data.experimental.cardinality(all_data).numpy()
train_size = int(dataset_size * (1 - validation_split - test_split))
validation_size = int(dataset_size * validation_split)

# Split the dataset into train, validation, and test sets
train_dataset = all_data.take(train_size)
remaining_data = all_data.skip(train_size)
validation_dataset = remaining_data.take(validation_size)
test_dataset = remaining_data.skip(validation_size)

# Initialize lists to store true labels and predictions
y_true = []
y_pred = []

# Iterate over the test dataset
for images, labels in test_dataset:
    # Get model predictions
    predictions = model.predict(images)
    # Convert predictions and labels to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels.numpy(), axis=1)
    # Append to lists
    y_pred.extend(predicted_classes)
    y_true.extend(true_classes)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(all_data.class_names)
# Generate the classification report
report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)

num_images = 16 # number of images to display
sample_images = train_df.sample(num_images)['image'].values # random sample of images


fig, axes = plt.subplots(4, 4, figsize=(12, 12))
plt.subplots_adjust(wspace=0.2, hspace=0.2) # spacing between images

"""for i, ax in enumerate(axes.flat):
    # Open the image file
    img = Image.open(sample_images[i])
    # Ensure the image is in RGB format
    ax.imshow(img) # display images
    # img = img.convert('RGB')
    # Resize the image to match the input size of the model
    target_size = (224, 224)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    true_label = train_labels[i]
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = all_data.class_names[predicted_class_index]
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}", fontsize=12) # set title
    ax.axis('off') # Remove axis labels for a cleaner look
plt.tight_layout()
plt.show()"""

def predict_and_display_images(model, dataset, class_names, num_images=16):
    # Select a batch of images from the test dataset
    image_batch, label_batch = next(iter(dataset.take(1)))  # Get a single batch of images
    image_batch = image_batch[:num_images]
    label_batch = label_batch[:num_images]
    
    # Make predictions on the batch
    predictions = model.predict(image_batch)
    predicted_classes = np.argmax(predictions, axis=1)
    real_classes = np.argmax(label_batch, axis=1)
    
    # Plot the images with predicted and true labels
    plt.figure(figsize=(16, 8))
    for i in range(num_images):
        ax = plt.subplot(2, 4, i + 1)
        img = array_to_img(image_batch[i])  # Convert back to a displayable image format
        plt.imshow(img)
        true_label = class_names[real_classes[i]]
        predicted_label = class_names[predicted_classes[i]]
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=12)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

predict_and_display_images(model, test_dataset, class_names=all_data.class_names, num_images=8)