import time
start = time.time()

import tensorflow as tf
import pandas as pd
import numpy as np
import os

# tf.logging.set_verbosity(tf.logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.ERROR)


# #when using Jupyter Notebook
# dir = os.getcwd()

# when using local runtime
dir = os.path.dirname(__file__)

train_url = 'https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv'

train_path = tf.keras.utils.get_file(
    os.path.join(dir, train_url.split('/')[-1]), train_url)

mnist_dataframe = pd.read_csv(train_path, sep=",", header=None)

# pd.options.display.float_format = '{:.1f}'.format
# mnist_dataframe.head()


# mnist_dataframe = mnist_dataframe.reindex(
#     np.random.permutation(mnist_dataframe.index))

# dataset = tf.convert_to_tensor(mnist_dataframe)

y_series = mnist_dataframe[0]
x_series = mnist_dataframe.loc[:, 1:784]
x_series /= 255
x_numpy = x_series.values
y_numpy = y_series.values
# x_train = x_numpy[:int(x_numpy.shape[0] * 0.1), :]
# x_test = x_numpy[int(x_numpy.shape[0] * 0.9):, :]
# y_train = y_numpy[:int(y_numpy.shape[0] * 0.1)]
# y_test = y_numpy[int(y_numpy.shape[0] * 0.9):]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_numpy, y_numpy)


# #Using Scikit-learn

# from sklearn.ensemble import RandomForestClassifier
# my_model = RandomForestClassifier().fit(x_train, y_train)

# # print(my_model.feature_importances_)
# print(my_model.predict([x_test[3434]]))
# # print(x_test[0])


# Using Tensorflow
with tf.Session() as sess:
    # Specify that all features have real-value data
    feature_name = "MNIST_features"
    feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                        shape=(1, 784))]
    # classifier = tf.estimator.LinearClassifier(
    #     feature_columns=feature_columns,
    #     n_classes=10,
    #     model_dir=os.path.join(dir, 'MNIST_models'))
    classifier = tf.estimator.DNNClassifier(
        hidden_units=[512],
        feature_columns=feature_columns,
        n_classes=10,
        dropout=0.2,
        model_dir=os.path.join(dir, 'MNIST_models'))

    # def input_fn(dataset):
    #     def _fn():
    #         features = {feature_name: tf.constant(dataset.data)} #dictionary
    #         label = tf.constant(dataset.target)
    #         return features, label
    #     return _fn

    def input_fn(x, y):
        def _fn():
            features = {feature_name: tf.constant(x)}
            label = tf.constant(y)
            return features, label
        return _fn

    # with tf.Session() as sess:
    #     print(sess.run(dataset.range(5)))
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     dataVar = tf.constant(y_train)
    #     print(dataVar.eval())
    # print(input_fn(x_train, y_train)())

    # # try train and eval fn with feature_spec
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=input_fn(x_train, y_train), max_steps=1000)
    # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn(x_test, y_test))

    # result = tf.estimator.train_and_evaluate(
    #     estimator=classifier,
    #     train_spec=train_spec,
    #     eval_spec=eval_spec
    # )
    # print(f'result = {result}')

    # writer = tf.summary.FileWriter(os.path.join(dir, 'log'))
    # writer.add_graph(sess.graph)
    # Fit model.
    with tf.name_scope("train"):
        classifier.train(input_fn=input_fn(x_train, y_train), steps=200)
    print('fit done')

    with tf.name_scope("evaluate"):
        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(
            input_fn=input_fn(x_test, y_test), steps=10)["accuracy"]
    print('\nAccuracy: {0:f}'.format(accuracy_score))

    # Export the model for serving
    feature_spec = {'MNIST_features': tf.FixedLenFeature(
        shape=(1, 784), dtype=np.float32)}

    serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)

# classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export',
#                                  serving_input_receiver_fn=serving_fn)
classifier.export_savedmodel(export_dir_base=os.path.join(dir, 'MNIST_models'),
                             serving_input_receiver_fn=serving_fn)


# # Using Keras
# import tensorflow.keras.callbacks as cb
# import tensorflow.keras.layers as layer


# def set_layer(name='set_layer'):
#     with tf.name_scope('layers'):
#         model = tf.keras.models.Sequential(layers=[
#             layer.Flatten(),
#             layer.Dense(512, input_shape=(784,), activation=tf.nn.relu),
#             layer.Dropout(0.2),
#             layer.Dense(10, activation=tf.nn.softmax)
#             # layer.Dense(10, activation='softmax')
#         ])
#     return model


# def compile_model(model, name='compile_model'):
#     with tf.name_scope('compile'):
#         model.compile(optimizer='Adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
#     return model


# dir = os.path.dirname(__file__)
# filepath = os.path.join(dir, 'MNIST_models/MNIST_callbacks.h5')
# log_dir = os.path.join(dir, 'log/1')
# # log_dir2 = os.path.join(dir, 'log/2')
# callbacks = [
#     cb.ModelCheckpoint(filepath=filepath, verbose=1,
#                        monitor='val_loss', save_best_only=True),
#     cb.TensorBoard(log_dir=log_dir, histogram_freq=2, write_grads=True,
#                    write_images=True),
#     cb.EarlyStopping(patience=1)
# ]


# def fit_model(model, callbacks, name='fit_model'):
#     with tf.name_scope('train'):
#         model.fit(x_train, y_train, epochs=20, validation_split=0.1,
#                   callbacks=callbacks)
#     return model


# model=fit_model(compile_model(set_layer()), callbacks)

# # model = tf.keras.models.load_model(filepath=filepath)

# test_result = model.evaluate(x=x_test, y=y_test)
# print(
#     f'Test result:\nTest loss = {test_result[0]}, Test accuracy = {test_result[1]}')


end = time.time()
print(f'Used time -- {end - start}')
