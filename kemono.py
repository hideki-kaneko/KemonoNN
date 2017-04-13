import os
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import KemonoNN as knn

IMG_SIZE = 28 #リサイズ後の画像の1辺の長さ
IMG_CH = 3 #色チャネル数
IMG_PIXELS = IMG_SIZE * IMG_SIZE * IMG_CH #総ピクセル数
TRAIN_PATH = "./data"       #学習用データセット
TEST_PATH = "./test"        #検証用データセット
OMAKE_PATH = "./omake"      #1枚ずつ追加で判定するときのデータセット

# 画像読み込み
# 一度画像を読み込むと、その時の配列は保存される
# Use SavedAray = True に設定すると、保存済みの配列から読み込む（高速）
labels = os.listdir(TRAIN_PATH)
labels_num = os.listdir(TRAIN_PATH).__len__()
(train_image, train_label) = knn.load_images(TRAIN_PATH, name="train", useSavedArray=True)
(test_image, test_label) = knn.load_images(TEST_PATH, name="test", useSavedArray=True)
(omake_image, omake_label) = knn.load_images(OMAKE_PATH, name="omake", useSavedArray=False)

#ニューラルネットワーク
x = tf.placeholder(tf.float32, shape=[None, IMG_PIXELS])
y_ = tf.placeholder(tf.float32, shape=[None, labels_num])

W = tf.Variable(tf.zeros([IMG_PIXELS, labels_num]))
b = tf.Variable(tf.zeros([labels_num]))

#1層目
L1_OUTPUT_CH = 32
W_conv1 = knn.weight_variable([5, 5, IMG_CH, L1_OUTPUT_CH]) #5x5パッチで32チャネルに出力
b_conv1 = knn.bias_variable([L1_OUTPUT_CH])
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, IMG_CH])
h_conv1 = tf.nn.relu(knn.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = knn.max_pool_2x2(h_conv1)

#2層目
L2_OUTPUT_CH = 64
W_conv2 = knn.weight_variable([5, 5, L1_OUTPUT_CH, L2_OUTPUT_CH])
b_conv2 = knn.bias_variable([L2_OUTPUT_CH])
h_conv2 = tf.nn.relu(knn.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = knn.max_pool_2x2(h_conv2)

#3層目
L3_OUTPUT_SIZE = 7
L3_NEURONS = 2048
W_fc1 = knn.weight_variable([L3_OUTPUT_SIZE * L3_OUTPUT_SIZE * L2_OUTPUT_CH, L3_NEURONS])
b_fc1 = knn.bias_variable([L3_NEURONS])
h_pool2_flat = tf.reshape(h_pool2, [-1, L3_OUTPUT_SIZE*L3_OUTPUT_SIZE*L2_OUTPUT_CH])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#4層目（読み出し層）
W_fc2 = knn.weight_variable([L3_NEURONS, labels_num])
b_fc2 = knn.bias_variable([labels_num])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#評価
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


DISABLE_LERNING = True #Trueにすると学習済みのパラメータを読み出す。初回はFalseにすること
loop_num = 1000 #学習回数
batch_size = 100 #1ループの学習でランダムで取り出すサンプルの数
loss = []
conv_im = []

with tf.Session() as sess:
    saver = tf.train.Saver()
    if (DISABLE_LERNING):
        saver.restore(sess, "vault/train.ckpt")
        conv_im = np.load("vault/conv_im.npy")
    else:
        sess.run(tf.global_variables_initializer())
        for i in range(loop_num):
            batch_mask = np.random.choice(train_image.shape[0], batch_size)
            x_batch = train_image[batch_mask]
            y_batch = train_label[batch_mask]
            loss.append(sess.run(cross_entropy, feed_dict={
                x: x_batch, y_: y_batch, keep_prob: 1.0
            }))
            if i < 10:
                conv = sess.run(h_conv1, feed_dict={
                    x: train_image,
                    y_: train_label,
                    keep_prob: 1.0
                })
                conv_im.append(conv)
            if i%100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    x: x_batch,
                    y_: y_batch,
                    keep_prob: 1.0
                })
                print("step {}, accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={
                x: x_batch,
                y_: y_batch,
                keep_prob: 0.5
            })
        saver.save(sess, "vault/train.ckpt")
        conv_im = np.asarray(conv_im)
        np.save("vault/conv_im.npy", conv_im)
        plt.plot(np.arange(loop_num), loss)
        plt.show()

    print("test accuracy {}".format(accuracy.eval(feed_dict={
        x:test_image, y_:test_label, keep_prob: 1.0
    })))

    #フィルター重み表示
    # L1_weight = sess.run(W_conv1, feed_dict={
    #     x: train_image,
    #     y_: train_label,
    #     keep_prob: 1.0
    # })
    # f, axarr = plt.subplots(4, 8)
    # for i in range(32):
    #     ax = axarr[int(i / 8)][i % 8]
    #     img = []
    #     for j in range(5):
    #         for k in range(5):
    #             img.append(L1_weight[j][k][0][i])
    #     img = np.asarray(img).reshape([5, 5])
    #     ax.imshow(img)

    #フィルター後画像表示
    # fig1, axarr = plt.subplots(4, 8)
    # ims = []
    # canvas = []
    # for t in range(10):
    #     for i in range(32):
    #         ax = axarr[int(i/8)][i%8]
    #         img = []
    #         for j in range(IMG_SIZE):
    #             for k in range(IMG_SIZE):
    #                 img.append(conv_im[t][5200][j][k][i])
    #         img = np.asarray(img).reshape([IMG_SIZE, IMG_SIZE])
    #         im = ax.imshow(img)
    #         canvas.append(im)
    #     ims.append(canvas)
    #     plt.savefig(str(t) + ".png")
    # plt.show()

    #1枚ずつ追加で画像判定
    (omake_index, omake_prob) = sess.run((tf.argmax(y_conv, 1), y_conv), feed_dict={
        x: omake_image,
        keep_prob: 1.0
    })
    for i, j in zip(omake_index, omake_prob):
        print(labels[i], ":", j)
    print(omake_index)

    #画像別確率表示
    kemono_prob = sess.run(y_conv, feed_dict={
        x: test_image,
        y_: test_label,
        keep_prob: 1.0
    })

#画像別確率表示
# fig2, axarr = plt.subplots(4, 5)
# for i in range(700,720):
#     ax = axarr[int((i-700)/5)][i%5]
#     ax.imshow(test_image[i].reshape(28,28,-1))
#     box = np.vstack([(kemono_prob[i] * 100).astype(np.int), np.asarray(labels)])
#     max_index = np.nanargmax(box[0])
#     label = box[1][max_index] + "(" + box[0][max_index] + "%),"
#     box_reduced = np.delete(box, max_index, 1)
#     max_index = np.nanargmax(box_reduced[0])
#     label += box_reduced[1][max_index] + "(" + box_reduced[0][max_index] + "%)"
#     ax.set_title(label)
# fig2.subplots_adjust(wspace=1.0, hspace=5.0)
#
# plt.figure()
# plt.imshow(train_image[5200].reshape([IMG_SIZE, IMG_SIZE, -1]))
# plt.show()