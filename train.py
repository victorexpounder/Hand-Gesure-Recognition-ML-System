import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
DATA_FILE = "gestures.csv"
data = np.loadtxt(DATA_FILE, delimiter=",")
X = data[:, 1:]  # Keypoints
y = data[:, 0]  # Labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("gesture_model.h5")
print("Model trained and saved!")
