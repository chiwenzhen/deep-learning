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
from pandas_confusion import ConfusionMatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

img_root = '3min-image-jpg/'
feature_data = 'feature_data.csv'


def save_feature():
    images = []
    labels = []
    app_names = []
    cates = [f for f in os.listdir(img_root)]
    for cate in cates:
        apps = [f for f in os.listdir(path.join(img_root, cate))]
        for app in apps:
            files = [f for f in os.listdir(path.join(img_root, cate, app)) if f.endswith('.jpg')]
            for f in files:
                images.append(path.join(img_root, cate, app, f))
                labels.append(cate)
                app_names.append(app)

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
    df = pd.DataFrame(features)
    df['label'] = labels
    df['app'] = app_names
    df.to_csv(feature_data, header=True, index=False)
    return df


if __name__ == '__main__':
    flag = 2
    # save_feature()
    df = pd.read_csv(path.join(feature_data), header=0)

    if flag == 1:
        features = df[df.columns.difference(['label', 'app'])].values
        labels = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test, y_pred) * 100))

        confusion_matrix = ConfusionMatrix(y_test, y_pred, display_sum=False)
        print(confusion_matrix)
    elif flag ==2:
        cates = ['video', 'live', 'audio', 'radio']
        for cate in cates:
            total_num = 0.0
            wright_num = 0
            accuracy_list = []
            apps = np.unique(df[df.label == cate]['app'].values)
            for i, app in enumerate(apps):
                df_train = df[df.app != app]
                df_test = df[df.app == app]
                x_train, y_train = df_train[df.columns.difference(['label', 'app'])].values, df_train['label'].values
                x_test, y_test = df_test[df.columns.difference(['label', 'app'])].values, df_test['label'].values
                clf = Pipeline([('std', StandardScaler()), ('clf', SVC())])
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                print('{}\t{:<30}\t{}'.format(datetime.datetime.now(), app, accuracy))
                accuracy_list.append(accuracy)
                total_num += len(y_test)
                wright_num += len(y_test) * accuracy
            print("-------------mean accuracy of {} :{}/{}={}".format(cate, wright_num, total_num,
                                                                      wright_num / total_num))
            print("-------------mean accuracy of {} :{}\n\n".format(cate, np.mean(accuracy_list)))
    else:
        pass