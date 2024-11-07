# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend(shadow='True')
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)
  # Show and save plot
  plt.savefig('model_results.png', dpi=120)
  plt.close()


def mae(y_test, y_pred):
  """
  Calculates mean absolute error between y_test and y_preds.
  """
  return tf.reduce_mean(tf.abs(y_test - y_pred))
  

def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.reduce_mean(tf.square(y_test - y_pred))


# Check TensorFlow version
print(tf.__version__)


# Create features
X = np.arange(-100, 100, 4)

# Create labels
y = np.arange(-90, 110, 4)


# Split data into train and test sets
N = 25
X_train = X[:N] # first 25 examples (80% of data)
y_train = y[:N]

X_test = X[N:] # last 10 examples (20% of data)
y_test = y[N:]


# Set random seed for reproducibility
tf.random.set_seed(1989)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),  # Adding input shape for the first layer
    tf.keras.layers.Dense(1)
    ])

# Compile the model
model.compile(loss='mae',  # 'mae' as a string to specify mean absolute error
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

# Fit the model
model.fit(X_train, y_train, epochs=100)


# Make predictions on the test set
y_preds = model.predict(X_test)
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_preds)

# Calculate model metrics
mae_1 = np.round(float(mae(y_test, y_preds.squeeze())), 2)
mse_1 = np.round(float(mse(y_test, y_preds.squeeze())), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
