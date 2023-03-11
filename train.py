import numpy as np
import tensorflow as tf
import valohai
 
my_parameters = {
   'max_steps': 300,
   'learning_rate': 0.001,
   'dropout': 0.2,
}
 
valohai.prepare(step="train-model", default_parameters=my_parameters)
 
input_path = 'mnist.npz'
with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
 
x_train, x_test = x_train / 255.0, x_test / 255.0
 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(my_parameters['dropout']),
    tf.keras.layers.Dense(10, activation='softmax')
])
 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_parameters['learning_rate']),
              loss=loss_fn,
              metrics=['accuracy'])
 
model.fit(x_train, y_train, epochs=5)
 
model.evaluate(x_test,  y_test, verbose=2)
 
output_path = 'model.h5'
model.save(output_path)