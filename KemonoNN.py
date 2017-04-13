import os
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pylab as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #標準偏差0.1の正規分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def load_images(path, IMG_SIZE = 28, isGray = False, useSavedArray = False, name = "train"):
    if(useSavedArray):
        print("Loading Images... [restore]")
        array_image = np.load("vault/" + name + "_image.npy")
        array_label = np.load("vault/" + name + "_label.npy")
        print("done. Loaded {} images.".format(array_image.shape[0]))
        return (array_image, array_label)
    else:
        print("Loading Images...")
        labels = os.listdir(path)
        labels_num = labels.__len__()
        train_image = []
        train_label = []
        for i, dir in enumerate(labels):
            files = os.listdir(path + "/" + dir)
            for file in files:
                img = Image.open(path + "/" + dir + "/" + file)
                """:type:PIL.PngImagePlugin.PngImageFile"""
                if(isGray):
                    img = img.convert("L")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                imgArray = np.asarray(img)
                train_image.append(imgArray.flatten() / 255.0)

                tmp = np.zeros(labels_num)
                tmp[i] = 1
                train_label.append(tmp)
        train_image = np.asarray(train_image)
        train_label = np.asarray(train_label)
        np.save("vault/" + name + "_image.npy", train_image)
        np.save("vault/" + name + "_label.npy", train_label)
        print("done. Loaded {} images.".format(train_image.shape[0]))
        return (train_image, train_label)

def get_minibatch(data, batch_size):
    batch_mask = np.random.choice(data.shape[0], batch_size)
    batch = data[batch_mask]
    return batch