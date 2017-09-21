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
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from PIL import Image

img_root = '3min-data-dpi/'
feature_data = '3min_dpi_feature.csv'

num_classes = 5
# define neural network model
def base_model():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=10, activation='relu', input_shape=(1024, 1)))
    model.add(Conv1D(64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


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

    # generate features
    images_num = len(images)
    features = np.empty((images_num, 1024))
    for i, img in enumerate(images):
        print('#{} {} {} ({}/{})\r'.format(i+1, datetime.datetime.now(), path.basename(img), i+1, images_num))
        im = Image.open(img)
        features[i] = np.array([p for p in im.getdata()])
    df = pd.DataFrame(features)
    df['label'] = labels
    df['app'] = app_names
    df.to_csv(feature_data, header=True, index=False)
    return df


if __name__ == '__main__':
    flag = 1
    #save_feature()
    df = pd.read_csv(path.join(feature_data), header=0)

    if flag == 1:
        features = df[df.columns.difference(['label', 'app'])].values
        labels = df['app'].values

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        # keras中使用Conv1D要将训练数据组织成（n_samples, n_features, 1）的格式
        #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
       # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # 定义模型
        #clf = KerasClassifier(build_fn=base_model, nb_epoch=10, batch_size=32)
        clf = LogisticRegression()
        # 训练和测试
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy={}'.format(accuracy))
        print("\n分类报告：")

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
                clf = LogisticRegression()
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