# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from deepctr.models import DeepFM
from deepctr.layers import FM, CrossNet
from tensorflow.keras.regularizers import l2 

# Assume Diversity Enhanced Interaction Network has a multi-head approach for diversity

class DiversityEnhancedInteractionNetwork(tf.keras.layers.Layer): 
def __init__(self, output_dim, num_heads=4): # Added num_heads for diversity 
    super(DiversityEnhancedInteractionNetwork, self).__init__() 
    self.output_dim = output_dim 
    self.num_heads = num_heads # Number of heads for diversity 
def build(self, input_shape): 
    self.kernels = [self.add_weight(name=f'kernel_{i}', 
                shape=(int(input_shape[-1]), self.output_dim), 
                initializer='uniform', 
                trainable=True) 
                for i in range(self.num_heads)] 
def call(self, inputs): 
    outputs = [tf.matmul(inputs, kernel) for kernel in self.kernels] 
    # Concatenating the outputs from different heads to enhance divers 
    return tf.concat(outputs, axis=-1)

# Load the dataset
# Change the directory to where the dataset is located
os.chdir('Path to the directory')
filename = 'Training_data.csv'
dataset = pd.read_csv(filename, encoding='latin1') 

# Data preprocessing
# Fill missing values with the column mean
dataset.fillna(dataset.mean(), inplace=True) 

# Define features and labels
Class_ID = dataset[['Target']]
X = dataset[[ 
'Vs30 (m/s)', 
'Soil Thickness (m)', 
'Landslide Probability', 
'Slope (°)', 
'Elevation (m)', 
'Distance from faut', 
'Hydrological Soil (Pixel value)', 
'Building Area (Sq. Feet)', 
'Building Height (Feet)', 
'Building Volume (Cubic Feet)', 
]]
Y = np.ravel(Class_ID) 

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Define input layer
input_layer = Input(shape=(X_train_scaled.shape[1],)) 

# Factorization Machine (FM) component
fm_out = Dense(1, activation=None)(input_layer)

# Main interaction component
main_interaction = CrossNet(5)(input_layer)

# Diversity Enhanced Interaction Network component
dein_out_1 = DiversityEnhancedInteractionNetwork(output_dim=128, num_heads=4)(input_layer)
dein_out_2 = DiversityEnhancedInteractionNetwork(output_dim=64, num_heads=3)(input_layer)
dein_out_3 = DiversityEnhancedInteractionNetwork(output_dim=32, num_heads=2)(input_layer)

# Combine the outputs of Diversity Enhanced Interaction Network components
dein_combined = Concatenate()([dein_out_1, dein_out_2, dein_out_3]) 

# Attention mechanism applied to the combined Diversity Enhanced Interaction Network Outputs
attention_out = tf.keras.layers.Attention()([dein_combined, dein_combined]

# Deep Neural Network (DNN) component
deep_out = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(
deep_out = BatchNormalization()(deep_out)
deep_out = Dropout(0.5)(deep_out)
deep_out = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(d
deep_out = BatchNormalization()(deep_out)
deep_out = Dropout(0.3)(deep_out)
deep_out = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(d
deep_out = BatchNormalization()(deep_out)
deep_out = Dropout(0.3)(deep_out) 

# Concatenate all components
concat_layer = Concatenate()([fm_out, attention_out, deep_out]) 

# Output layer
output_layer = Dense(1, activation='sigmoid')(concat_layer) 

# Define the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer) 

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy, metrics=['accuracy'])
model.summary() 

# Train the model
history = model.fit(X_train_scaled, Y_train, epochs=500, batch_size=32, validation_split=0.3) 

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, Y_test)
print(f'Test Accuracy: {test_accuracy}') 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Step 1: Predict probabilities on the test set
# Assumes binary classification (adjust if multi-class)
y_scores = model.predict(X_test_scaled).ravel()  # Flatten to 1D
y_true = Y_test.ravel()  # Ensure it's also 1D

# Step 2: Compute precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Step 3: Compute FDR = 1 - precision
fdr = 1 - precision

# Step 4: Plot Precision-Recall Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, marker='.', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# Step 5: Plot False Discovery Rate vs Threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds, fdr[:-1], color='red', label='FDR = 1 - Precision')
plt.xlabel('Decision Threshold')
plt.ylabel('False Discovery Rate (FDR)')
plt.title('FDR vs Threshold')
plt.legend()

plt.tight_layout()
plt.show()

# Step 6: Print Average Precision Score
ap = average_precision_score(y_true, y_scores)
print(f'Average Precision (AP) Score: {ap:.4f}')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show() 

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show() 

# Predict the labels on the test set
Y_pred = model.predict(X_test_scaled)
Y_pred_classes = (Y_pred > 0.5).astype(int) 

# Generate the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_classes) 

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show() 

# Generate the classification report
class_report = classification_report(Y_test, Y_pred_classes)
print('Classification Report:')
print(class_report) 

# Load new dataset from another study area
filename_new = 'New_Data.csv'
new_dataset = pd.read_csv(filename_new, encoding='latin1')
new_dataset.fillna(new_dataset.mean(), inplace=True) 

# Define new features for the new dataset
X_new = new_dataset[[ 
'Vs30 (m/s)', 
'Soil Thickness (m)', 
'Landslide Probability', 
'Slope (°)', 
'Elevation (m)', 
'Distance from faut', 
'Hydrological Soil (Pixel value)', 
'Building Area (Sq. Feet)', 
'Building Height (Feet)', 
'Building Volume (Cubic Feet)'
]] 

# Standardize new features using the same scaler
X_new_scaled = scaler.transform(X_new) 

# Predict the labels for the new dataset
Y_new_pred = model.predict(X_new_scaled)
Y_new_pred_classes = (Y_new_pred > 0.5).astype(int) 

# Output the predictions for the new dataset
new_dataset['New_Predictions'] = Y_new_pred_classes
print(new_dataset[['New_Predictions']]) 

# Save the predictions to a new CSV file
new_dataset.to_csv('New_study_area_predictions.csv', index=False)

#SHAP analysis using KernelExplainer 
def model_predict(X):
    return model.predict(X, verbose=0).astype(np.float64)  

# Ensure NumPy array output
import shap
explainer = shap.KernelExplainer(model_predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:100])


# Extract the correct SHAP values for binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Convert to SHAP Explanation object
shap_values_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value[0], data=X_test[:100])


import shap
import matplotlib.pyplot as plt

# Set high-quality figure parameters
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
})

# Create a large figure with high DPI
fig = plt.figure(figsize=(12, 8), dpi=600)

# Plot SHAP beeswarm
shap.plots.beeswarm(shap_values_exp, show=False)  
# Avoid automatic show to modify further

# Customize axes (e.g., tick parameters)
plt.tick_params(axis='both', which='major', width=1.5, length=6)

# Save or display
plt.tight_layout()
plt.savefig("shap_beeswarm_high_quality.png", dpi=300, bbox_inches='tight')
plt.show()
