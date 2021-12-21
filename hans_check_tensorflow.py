import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import InputLayer
print(tf.config.list_physical_devices("GPU"))


x = tf.constant([[1, 2, 3, 4.5]])
y = x*tf.ones([2000, 1])
tf.print(y)


from tensorflow.python.keras.api import keras

input_layer = keras.layers.Input(3)
x = input_layer
x = keras.layers.Dense(4)(x)
x = tf.expand_dims(x, 2)
x = keras.layers.Conv1D(1, 2, 1, 'valid')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1)(x)
model = keras.Model(input_layer, x)

model.summary()

model.compile('adam', 'mse')

print(model(tf.constant([[1, 2, 3]])))