# coding=utf-8

import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pickle

image_dir = 'mnist/'

if not os.path.exists(image_dir+'mnist-test-data.csv'):
    # read images and labels
    df = pd.read_csv(image_dir + 'test-labels.csv', header=None)
    images = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values
    images_num = len(images)

    features = np.empty((images_num, 2048))

    # load model
    with gfile.FastGFile('inception-2015-12-05/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # generate features
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for i, img in enumerate(images):
            print('{}/{}\r'.format(i+1, images_num))
            image_data = gfile.FastGFile(image_dir+img, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[i, :] = np.squeeze(predictions)

    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(image_dir+'mnist-test-data.csv', header=True, index=False)
else:
    df = pd.read_csv(image_dir+'mnist-test-data.csv', header=0)
    features = df[df.columns.difference(['label'])].values
    labels = df['label'].values

# predict
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))
