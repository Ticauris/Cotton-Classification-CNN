import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

print("Script is starting...")


################# loading function ################
def load_images_and_labels(file_path, target_size=(256, 256)):
    if not os.path.isdir(file_path):
        raise FileNotFoundError(f"Directory {file_path} not found. Please check the path.")
    print(f"Loading images from {file_path}... \n")
    valid_extensions = [ d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
    print(f"Found {len(valid_extensions)} valid classes: {valid_extensions} \n")
    valid_extensions = sorted(valid_extensions)
    label_dictionary = {label: index for index, label in enumerate(valid_extensions)}
    print(f"Label Dictionary: {label_dictionary} \n")
    images = []
    labels = []

    for label in valid_extensions:
        label_path = os.path.join(file_path, label)
        files = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            images.append(image)
            labels.append(label_dictionary[label])

    print(f"Loaded {len(images)} images from {file_path}")
    return np.array(images), np.array(labels)


'''################# cal green index ###################
def calculate_green_index(images, threshold=0.65):
    print("Calculating Green Index...")
    green_index_images = []
    green_index_labels = []
    for image in images:
        # Extract individual bands
        red_band = image[:, :, 0].astype("float32")
        green_band = image[:, :, 1].astype("float32")
        blue_band = image[:, :, 2].astype("float32")
        # Calculate Green Index
        green_index = ((2 * green_band) - red_band - blue_band) / (
            (2 * green_band) + red_band + blue_band + 1e-10
        )
        green_index_images.append(green_index)
        green_index_labels.append(int(np.mean(green_index) > threshold))
        """if np.mean(green_index) > threshold:
            green_index_labels.append(1)
        else:
            green_index_labels.append(0)"""
    return np.array(green_index_images), np.array(green_index_labels).astype(int)'''


################## filter images ####################
def filter_images(image_dir, extensions=(".jpg", ".jpeg", ".png")):
    print(f"Filtering non-image files in directory: {image_dir} \n")
    for root, dirs, files in os.walk(image_dir, topdown=True):
        for file_name in files:
            if not file_name.lower().endswith(extensions):
                if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"Removing non-image file: {file_name} \n")
                    file_path = os.path.join(root, file_name)
                    os.remove(file_path)


################## display images and labels ####################
def rgb_with_img_and_label_grid(
    image, label, titles=None, rows=4, cols=4, cmap="viridis"
):
    if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
        raise ValueError("Images and labels should be numpy arrays.")

    num_images, num_labels = image.shape[0], label.shape[0]
    if num_images != num_labels:
        raise ValueError("The number of images and labels must be the same.")

    num_plots = rows * cols
    if num_images < num_plots:
        raise ValueError(
            f"Not enough images/labels to fill the subplots: {num_plots} plots expected but got {num_images} images/labels."
        )

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))
    axes = axes.flatten()

    for ax, img, lbl in zip(axes, image, label):
        ax.imshow(img, cmap=cmap)
        class_label = np.argmax(lbl)
        ax.set_title(f"Label: {class_label}", color="black")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


################# augment data ###################
"""def data_augmentation():
    print("Configuring data augmentation...")
    # Create an instance of the ImageDataGenerator with desired augmentations
    return ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.5, 1.5),
        fill_mode="nearest",
        rescale=1.0 / 255,  # Rescale pixel values
    )"""


#################### U-Net model ####################
def unet(input_shape, number_of_classes=4):
    # Define the convolutional block function
    def conv_block(
        inputs, filters, kernel_size=(3, 3), activation="relu", padding="same"
    ):
        conv = layers.Conv2D(
            filters, kernel_size, activation=activation, padding=padding
        )(inputs)
        conv = layers.Conv2D(
            filters, kernel_size, activation=activation, padding=padding
        )(conv)
        return conv

    inputs = tf.keras.Input(shape=input_shape)
    # Contracting Path
    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 1024)

    gap = layers.GlobalAveragePooling2D()(conv5)

    # Expanding Path
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv5)
    concat6 = layers.concatenate([up6, conv4], axis=3)
    conv6 = conv_block(concat6, 512)
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv6)
    concat7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = conv_block(concat7, 256)
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv7)
    concat8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = conv_block(concat8, 128)
    flatten = layers.Flatten()(gap)

    # Output layer
    outputs = layers.Dense(number_of_classes, activation="softmax")(
        flatten
    )  # softmax activation function

    # Model definition
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def expand_labels(labels):
    labels = np.expand_dims(labels, axis=-1)  # Add one dimension
    labels = np.expand_dims(labels, axis=-1)  # Add another dimension
    labels = np.expand_dims(labels, axis=-1)
    return labels


################### data loading and filtering ####################
# define directories
test_dir = "/Users/ticaurisstokes/Desktop/Cotton CNN/Cotton Disease/test"
train_dir = "/Users/ticaurisstokes/Desktop/Cotton CNN/Cotton Disease/train"
val_dir = "/Users/ticaurisstokes/Desktop/Cotton CNN/Cotton Disease/val"

# filter directories
filter_images(test_dir)
filter_images(train_dir)
filter_images(val_dir)

# load images and labels
test_images, test_labels = load_images_and_labels(test_dir)
train_images, train_labels = load_images_and_labels(train_dir)
val_images, val_labels = load_images_and_labels(val_dir)

test_labels = expand_labels(test_labels)
train_labels = expand_labels(train_labels)
val_labels = expand_labels(val_labels)

print(f"Expanded Test Labels: {test_labels.shape}")
print(f"Expanded Train Labels: {train_labels.shape}")
print(f"Expanded Validation Labels: {val_labels.shape}")

"""# check if images are loaded
print(f"Test Images: {test_images.shape}")
print(f"Train Images: {train_images.shape}")
print(f"Validation Images: {val_images.shape}")"""


############################### Expand labels #################################
# convert labels to one-hot encoding
number_of_classes = 4
test_labels = to_categorical(test_labels, num_classes=number_of_classes)
train_labels = to_categorical(train_labels, num_classes=number_of_classes)
val_labels = to_categorical(val_labels, num_classes=number_of_classes)

# Reshape labels to remove singleton dimensions
test_labels = np.squeeze(test_labels, axis=(1, 2))
train_labels = np.squeeze(train_labels, axis=(1, 2))
val_labels = np.squeeze(val_labels, axis=(1, 2))

# check if labels are converted
print(f"Test Labels: {test_labels.shape}\n")
print(f"Train Labels: {train_labels.shape}\n")
print(f"Validation Labels: {val_labels.shape}\n")

############################### Display RGB and Unity Images and Labels ##########################
"""rgb_with_img_and_label_grid(test_images, test_labels, rows=4, cols=4)
rgb_with_img_and_label_grid(train_images, train_labels, rows=4, cols=4)
rgb_with_img_and_label_grid(val_images, val_labels, rows=4, cols=4)"""

############################### Data Augmentation ############################
# Augment the data
'''data_augmentor = data_augmentation()
augmented_data = data_augmentor.flow(train_images, train_labels, batch_size=32)
# Display augmented images
augmented_images, augmented_labels = next(augmented_data)'''

# Convert to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(32)
print(f"Train images shape: {train_images.shape}\n")
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(16).prefetch(-1)
print(f"Value images shape: {val_images.shape}\n")
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(16).prefetch(-1)
print(f"Test images shape: {test_images.shape}\n")
"""
print(f"Total batches in training data: {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"Total batches in validation data: {tf.data.experimental.cardinality(val_dataset).numpy()}")
print(f"Total batches in test data: {tf.data.experimental.cardinality(test_dataset).numpy()}")
"""

"""print(f"Shape of augmented images: {augmented_images.shape}\n")
print(f"Shape of augmented labels: {augmented_labels.shape}\n")"""

print(f"model input shape: {train_images.shape[1:], val_dataset.element_spec[0].shape}")

# Initialize the model
model = unet((256, 256, 3), number_of_classes=4)
######################################## Compile the model  ################################
print("Compiling Model...")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# Define callbacks
model_checkpoint = ModelCheckpoint(
    "unet_model.keras", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
)

############################## Train the model #####################################

print("Training Model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[model_checkpoint, early_stopping],
)

######################## Evaluate the model on the test set  ###########################
# Extract metrics from the training history
print("Evaluating Model...")
test_loss, test_accuracy = model.evaluate(test_dataset.batch(32))
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(accuracy, label="Training Accuracy", color="blue")
plt.plot(val_accuracy, label="Validation Accuracy", color="orange")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(loss, label="Training Loss", color="blue")
plt.plot(val_loss, label="Validation Loss", color="orange")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Print training and validation loss
print("Training Loss:", loss)
print("Validation Loss:", val_loss)
