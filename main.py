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

# Load the dataset
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

# Create the DataFrame with the image paths and corresponding labels
train_df = pd.DataFrame({
    'image': train_files,
    'label': train_labels
})

print(train_df.head())
print(train_df['label'].value_counts())

num_images = 16 # number of images to display
sample_images = train_df.sample(num_images)['image'].values # random sample of images


"""fig, axes = plt.subplots(4, 4, figsize=(12, 12))
plt.subplots_adjust(wspace=0.2, hspace=0.2) # spacing between images

for i, ax in enumerate(axes.flat):
    # Open the image file
    img = Image.open(sample_images[i])
    print(img.size)
    ax.imshow(img) # display images
    ax.set_title(train_labels[i]) # set title
    ax.axis('off') # Remove axis labels for a cleaner look
plt.tight_layout()
plt.show()"""

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

# data-splitting
# Get the size of the complete dataset
dataset_size = tf.data.experimental.cardinality(all_data).numpy()

# Get the sizes of the datasets
train_size = tf.data.experimental.cardinality(train_dataset).numpy()
validation_size = tf.data.experimental.cardinality(validation_dataset).numpy()
test_size = tf.data.experimental.cardinality(test_dataset).numpy()

# Calculate the proportions
train_proportion = train_size / dataset_size
validation_proportion = validation_size / dataset_size
test_proportion = test_size / dataset_size

# proportion
print(f"Train dataset proportion: {train_proportion:.2%}")
print(f"Validation dataset proportion: {validation_proportion:.2%}")
print(f"Test dataset proportion: {test_proportion:.2%}")

base_model = NASNetMobile(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3))

# Congela i pesi del modello preaddestrato
for layer in base_model.layers:
    layer.trainable = False

# Aggiungi strati personalizzati sopra il modello preaddestrato
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)  # Cambia l'output a 2 classi con softmax

model = Model(inputs=base_model.input, outputs=output)

# Compila il modello con categorical_crossentropy per multi-class classification
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Aggiungi EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Addestramento con EarlyStopping
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
    shuffle=False
)

# Salva il modello
model.save('rmi_braintumor_classifier.h5')

# Visualizza le performance
import matplotlib.pyplot as plt

# Estrai le perdite di training e validazione
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Estrai l'accuratezza di training e validazione
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Grafico delle perdite
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Grafico dell'accuratezza
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
