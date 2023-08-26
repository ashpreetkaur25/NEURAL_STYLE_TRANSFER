"""Importing the libraries"""
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from lost_function import loss
from tensorflow.keras.applications import VGG19

"""This file performs the style transfer algorithm to extract the features of the content image
    and the style of the style image and combine them to generate a new stylised image"""

"""A function to change the size of the image"""


def change_size(image):
    image = plt.imread(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [400, 400])
    image = image[tf.newaxis, :]
    return image


def extract_image(style_image):
    """Importing the images"""
    content = plt.imread("Golden gate.jpg")
    style = plt.imread(style_image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    ax1.imshow(content)
    ax1.set_title("Content image")
    ax2.imshow(style)
    ax2.set_title("Chosen Style image")
    plt.show()

    """Resizing the images"""
    new_content1 = change_size("Golden gate.jpg")
    new_style1 = change_size(style_image)

    return new_content1, new_style1


def display_image(img, training_step):
    if len(list(np.shape(img))) > 3:
        plt.figure(figsize=(6, 4))
        plt.imshow(np.squeeze(img.read_value(), 0))
        plt.title(f"Training step{training_step}")
        plt.imsave("stylised_image.jpg", img)
        plt.show()


"""A function to train the style transfer algorithm"""


@tf.function()
def train_style_transfer(image, extracted_content_features, extracted_style_features, tv_weight):
    """Using GradientTape class to compute the gradients to minimize the loss"""

    with tf.GradientTape() as tape:
        outputs = extractor.features(image)
        cost = loss(extracted_content_features, extracted_style_features, outputs)
        cost += tv_weight * tf.image.total_variation(image)

    grad = tape.gradient(cost, image)

    opt.apply_gradients([(grad, image)])

    return image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


if __name__ == '__main__':
    new_image = tf.Variable(change_size("Golden gate.jpg"))

    """Initialising the VGG-19 model"""
    vgg = VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    """Computing the style and the content layers """
    for layers in vgg.layers:
        print(layers.name)

    """Style and content layer"""
    content_layers = ['block5_conv2']
    style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    """Setting the optimizer and FeatureExtractor class object"""
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    extractor = FeatureExtractor(content_layers, style_layers, vgg)

    """Extracting content and style features by user's choice"""

    print("Following are the options for styles...")

    style1 = plt.imread("style-1.jpg")
    style2 = plt.imread("style-2.jpg")
    style3 = plt.imread("style-3.jpg")

    rows = 1
    column = 3

    fig1 = plt.figure(figsize=(6, 4))
    fig1.add_subplot(rows, column, 1)
    plt.imshow(style1)
    plt.axis("off")
    plt.title("Style-1")

    fig1.add_subplot(rows, column, 2)
    plt.imshow(style2)
    plt.axis("off")
    plt.title("Style-2")

    fig1.add_subplot(rows, column, 3)
    plt.imshow(style3)
    plt.axis("off")
    plt.title("Style-3")
    plt.show()

    """Asking the user for the style he/she want to choose"""

    choice = input("Enter which style do you wanna go for-\n")

    if choice == "style1":
        new_content, new_style = extract_image("style-1.jpg")
        content_features = extractor.features(new_content)["content"]
        style_features = extractor.features(new_style)["style"]

    elif choice == "style2":
        new_content, new_style = extract_image("style-2.jpg")
        content_features = extractor.features(new_content)["content"]
        style_features = extractor.features(new_style)["style"]

    elif choice == "style3":
        new_content, new_style = extract_image("style-3.jpg")
        content_features = extractor.features(new_content)["content"]
        style_features = extractor.features(new_style)["style"]

    else:
        flag = 0
        style_image1 = f"{choice}.jpg"
        for images in os.listdir("./"):
            if style_image1 == images:
                flag = 1
                new_content, new_style = extract_image(style_image1)
                content_features = extractor.features(new_content)["content"]
                style_features = extractor.features(new_style)["style"]
                break
        if flag == 0:
            print("No such style available\n")
            exit(0)

    """Training the style transfer algorithm"""

    epochs = 15
    step_per_epochs = 100
    steps = 0
    start = time.time()

    for i in range(epochs):
        for j in range(step_per_epochs):
            steps += 1
            train_style_transfer(new_image, content_features, style_features, 30)
        if steps % 500 == 0:
            display_image(new_image, steps)
        print(f"{i+1}. {steps} training steps completed")
    print("Image stylised and saved successfully..")

    end = time.time()
    print(f"Time taken to stylised the image: {end-start}")
