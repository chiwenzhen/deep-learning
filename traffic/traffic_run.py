# coding=utf-8
"""
利用已经训练好的inception，从mnist图片提取特征（2048-byte vector），并用SVM进行分类
"""

import os
import os.path as path
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

data_dir = 'data/'
img_root = '3min-image-jpg/'
feature_data = 'feature_data.csv'

# load image paths and labels
if not path.exists(path.join(data_dir, feature_data)):
    images = []
    labels = []
    cates = [f for f in os.listdir(img_root)]
    for cate in cates:
        apps = [f for f in os.listdir(path.join(img_root, cate))]
        for app in apps:
            files = [f for f in os.listdir(path.join(img_root, cate, app)) if f.endswith('.jpg')]
            for f in files:
                images.append(path.join(img_root, cate, app, f))
                labels.append(cate)

    # load model
    with gfile.FastGFile('../inception-2015-12-05/classify_image_graph_def.pb', 'rb') as img:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(img.read())
        _ = tf.import_graph_def(graph_def, name='')

    # generate features
    images_num = len(images)
    features = np.empty((images_num, 2048))
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for i, img in enumerate(images):
            print('#{} {} {} ({}/{})\r'.format(i+1, datetime.datetime.now(), path.basename(img), i+1, images_num))
            image_data = gfile.FastGFile(img, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(predictions)
    if not path.exists(data_dir):
        os.makedirs(data_dir)
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(path.join(data_dir, feature_data), header=True, index=False)
else:
    df = pd.read_csv(path.join(data_dir, feature_data), header=0)
    features = df[df.columns.difference(['label'])].values
    labels = df['label'].values

# predict
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
