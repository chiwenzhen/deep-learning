# coding=utf-8
"""
将mnist图像看成一维向量，利用1维卷积核
"""
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist

batch_size = 128
num_classes = 10
epochs = 12


# define neural network model
def base_model():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=10, activation='relu', input_shape=(784, 1)))
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


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # keras中使用Conv1D要将训练数据组织成（n_samples, n_features, 1）的格式
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2], 1)

    # 定义模型
    clf = KerasClassifier(build_fn=base_model, nb_epoch=12, batch_size=128)
    # 训练和测试
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy={}'.format(accuracy))
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
