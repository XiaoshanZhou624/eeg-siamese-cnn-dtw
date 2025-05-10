import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, Lambda, Dropout, BatchNormalization,
                                     Multiply, Add, Activation, GlobalAveragePooling1D, Reshape, LayerNormalization, MultiHeadAttention)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import shap

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Load EEG Data
df = pd.read_csv('/..')

# Define parameters
n = 24  # Number of time steps per sequence
feature_columns = ['POW.AF4.Theta', 'POW.AF4.Gamma', 'POW.AF3.Alpha', 'POW.FC6.BetaH']  # Multiple features
num_features = len(feature_columns)

# Group data into sequences
grouped = [df[i:i+n] for i in range(0, len(df), n)]

# Extract multiple features
X = np.array([[group[col].values for col in feature_columns] for group in grouped]).transpose(0, 2, 1)
y = np.array([group['marker_value'].iloc[0] for group in grouped])

# Scale time series data
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
X_scaled = scaler.fit_transform(X)  # Directly normalize across all samples and features

# Function to create pairs
def create_pairs(X, y):
    positive_pairs = []
    negative_pairs = []

    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                if y[i] == y[j]:
                    positive_pairs.append([X[i], X[j], 1])
                else:
                    negative_pairs.append([X[i], X[j], 0])

    pairs = positive_pairs + negative_pairs
    np.random.shuffle(pairs)
    pairs = np.array(pairs, dtype=object)

    return np.stack(pairs[:, 0]), np.stack(pairs[:, 1]), np.array(pairs[:, 2], dtype='float32')

def evaluate_model(model, X1_test, X2_test, y_test):
    """Evaluate the trained model on test data."""
    y_pred_prob = model.predict([X1_test, X2_test])  # Get probabilities
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert to binary predictions

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    return {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1-score": f1, "AUC": auc}

# ** Define the Siamese network architecture **
def create_base_network_cnn(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    return Model(inputs=input_layer, outputs=x)


# Network setup
input_shape = (n, num_features)
base_network = create_base_network_cnn(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
outputs = Dense(1, activation='sigmoid')(distance)

model = Model(inputs=[input_a, input_b], outputs=outputs)

# Compile the model
# optimizer = Adam(learning_rate=0.001)
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Prepare data
X1, X2, labels = create_pairs(X_scaled, y)
X1 = np.stack(X1).astype('float32')
X2 = np.stack(X2).astype('float32')

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('/best_model.h5', monitor='val_loss', save_best_only=True)

# Fit model
history = model.fit([X1, X2], labels, validation_split=0.2, epochs=100, callbacks=[early_stopping, checkpoint])

# ** Integrated Gradients and Attention Weights Visualization **
def interpolate_inputs(baseline, input_tensor, steps=50):
    """Generate interpolated inputs between the baseline and input."""
    alphas = np.linspace(0, 1, steps + 1).reshape(-1, 1, 1)  # Create a range of alpha values
    interpolated_inputs = baseline + alphas[:, np.newaxis, :, :] * (input_tensor - baseline)  # Shape (steps+1, 24, 4)
    return interpolated_inputs

@tf.function
def compute_gradients(model, inputs):
    """Compute gradients of model output w.r.t. inputs."""
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)  # Ensure TensorFlow dtype
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)  # Ensure model is not in training mode
    return tape.gradient(predictions, inputs)

def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    """Compute Integrated Gradients using TensorFlow best practices."""
    if baseline is None:
        baseline = np.zeros_like(input_tensor)  # Baseline should match input shape

    # Generate interpolated inputs
    alphas = np.linspace(0, 1, steps + 1).reshape(-1, 1, 1, 1)  # Ensure correct broadcasting
    interpolated_inputs = baseline + alphas * (input_tensor - baseline)  # Shape (steps+1, 1, 24, 4)

    # Compute gradients at each interpolation step
    grads = np.array([
        compute_gradients(model, tf.convert_to_tensor(inp, dtype=tf.float32)).numpy()
        for inp in interpolated_inputs
    ])

    # Average gradients
    avg_grads = np.mean(grads, axis=0)

    # Compute Integrated Gradients
    ig_values = (input_tensor - baseline) * avg_grads
    return ig_values

# Select a sample input from your test data (ensure batch dimension is added)
sample_input = np.expand_dims(X_scaled[0], axis=0).astype(np.float32)  # Shape (1, 24, 4)
baseline = np.zeros_like(sample_input)  # Baseline should match sample input shape


# Compute Integrated Gradients
ig_values = integrated_gradients(base_network, sample_input)
print(ig_values)

# Plot Integrated Gradients
plt.figure(figsize=(10, 5))
plt.plot(ig_values.squeeze())  # Flatten for visualization
plt.title("Integrated Gradients for EEG Features")
plt.xlabel("Time Steps")
plt.ylabel("Gradient Importance")
plt.grid(True)
plt.legend(feature_columns, loc='upper right')
# save the image
plt.savefig("integrated_gradients_attention.png", dpi=600, bbox_inches='tight')
plt.show()

