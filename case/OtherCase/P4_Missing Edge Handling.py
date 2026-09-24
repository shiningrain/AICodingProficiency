import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Demo input data
x_data = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
y_data = [6, 7, 8]

# Convert to numpy arrays
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_data_reshaped = x_data.reshape(-1, 1)
y_data_reshaped = y_data.reshape(-1, 1)

x_normalized = scaler_x.fit_transform(x_data_reshaped).reshape(x_data.shape)
y_normalized = scaler_y.fit_transform(y_data_reshaped).flatten()

# Reshape for RNN input (samples, timesteps, features)
x_train = x_normalized.reshape(x_normalized.shape[0], x_normalized.shape[1], 1)
y_train = y_normalized

# Build the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(5, 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=150, batch_size=1, verbose=1)

# Make predictions
predictions = model.predict(x_train)
predictions_denormalized = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

print("\nOriginal sequences:")
for i, seq in enumerate(x_data):
    print(f"Sequence {i+1}: {seq} -> Actual: {y_data[i]}, Predicted: {predictions_denormalized[i]:.2f}")

# Test with a new sequence
test_sequence = np.array([[4, 5, 6, 7, 8]], dtype=np.float32)
test_normalized = scaler_x.transform(test_sequence.reshape(-1, 1)).reshape(test_sequence.shape)
test_reshaped = test_normalized.reshape(1, 5, 1)
test_prediction = model.predict(test_reshaped)
test_prediction_denormalized = scaler_y.inverse_transform(test_prediction.reshape(-1, 1)).flatten()

print(f"\nTest sequence: {test_sequence[0]} -> Predicted next value: {test_prediction_denormalized[0]:.2f}")