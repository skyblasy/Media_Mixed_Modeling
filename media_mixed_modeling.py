import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Reshape
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('media_data.csv')
df['WEEK_START'] = pd.to_datetime(df['WEEK_START'])

# Create geo dummies
geo_dummies = pd.get_dummies(df['GEO'], prefix='geo')

# Define feature sets
spend_features = ['GOOGLE_SPEND', 'FB_SPEND', 'AFF_SPEND']
impression_features = ['YT_PAID_IMP', 'YT_ORG_IMP', 'GOOGLE_IMP', 'EMAIL_IMP', 'FB_IMP', 'AFF_IMP']

# Prepare input features
X_spend = pd.concat([df[spend_features], geo_dummies], axis=1)
X_impressions = df[impression_features]
y_revenue = df['REVENUE']

# Scale the features
scaler_spend = StandardScaler()
scaler_imp = StandardScaler()
scaler_rev = StandardScaler()

X_spend_scaled = scaler_spend.fit_transform(X_spend)
X_imp_scaled = scaler_imp.fit_transform(X_impressions)
y_revenue_scaled = scaler_rev.fit_transform(y_revenue.values.reshape(-1, 1))

# Create sequences for LSTM
seq_length = 4
X_spend_seq = []
X_imp_seq = []
y_seq = []

for i in range(len(X_spend_scaled) - seq_length):
    X_spend_seq.append(X_spend_scaled[i:i+seq_length])
    X_imp_seq.append(X_imp_scaled[i:i+seq_length])
    y_seq.append(y_revenue_scaled[i+seq_length])

X_spend_seq = np.array(X_spend_seq)
X_imp_seq = np.array(X_imp_seq)
y_seq = np.array(y_seq)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_spend_seq, y_seq, test_size=0.2, random_state=42
)

# Build the model with corrected architecture
spend_input = Input(shape=(seq_length, X_spend_seq.shape[2]), name='spend_input')

# Stage 1: Spend -> Impressions (maintaining temporal dimension)
x1 = Dense(64, activation='relu')(spend_input)
x1 = Dense(32, activation='relu')(x1)
impressions_output = Dense(len(impression_features), activation='linear', name='impressions_output')(x1)

# Stage 2: Impressions -> Revenue
# Now impressions_output has shape (batch, seq_length, num_impressions)
lstm_out = LSTM(64, return_sequences=True)(impressions_output)
revenue_intermediate = Dense(32, activation='relu')(lstm_out)

# Stage 3: Combined Spend & Impressions -> Revenue
# Flatten the temporal dimension for the final prediction
combined = Concatenate(axis=2)([spend_input, revenue_intermediate])
x3 = LSTM(64, return_sequences=False)(combined)  # Remove temporal dimension
x3 = Dense(32, activation='relu')(x3)
revenue_output = Dense(1, activation='linear', name='revenue_output')(x3)

# Create and compile model
model = Model(inputs=spend_input, outputs=[impressions_output, revenue_output])
model.compile(
    optimizer='adam',
    loss={
        'impressions_output': 'mse',
        'revenue_output': 'mse'
    },
    loss_weights={
        'impressions_output': 0.3,
        'revenue_output': 0.7
    }
)

# Train the model
history = model.fit(
    X_train,
    {
        'impressions_output': X_imp_seq[len(X_imp_seq)-len(X_train):],
        'revenue_output': y_train
    },
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Make predictions
predictions = model.predict(X_test)
impression_preds = predictions[0]
revenue_preds = predictions[1]

# Inverse transform predictions to original scale
revenue_preds_original = scaler_rev.inverse_transform(revenue_preds)
impression_preds_original = scaler_imp.inverse_transform(impression_preds[:, -1, :])  # Take last timestep

# Calculate spend coefficients
spend_coefficients = {}
for i, feature in enumerate(spend_features):
    base_input = X_test[0:1].copy()
    increased_input = X_test[0:1].copy()
    increased_input[0, :, i] += 0.1  # Small increase for better gradient approximation
    
    base_pred = model.predict(base_input)[1]
    increased_pred = model.predict(increased_input)[1]
    
    coefficient = ((increased_pred - base_pred)[0][0]) * 10  # Scale back up
    spend_coefficients[feature] = coefficient

print("\nSpend Coefficients (Revenue change per unit spend):")
for feature, coef in spend_coefficients.items():
    print(f"{feature}: {coef:.4f}")

# Calculate geo effects
geo_cols = [col for col in X_spend.columns if col.startswith('geo_')]
geo_coefficients = {}

for i, geo in enumerate(geo_cols):
    idx = len(spend_features) + i
    base_input = X_test[0:1].copy()
    increased_input = X_test[0:1].copy()
    increased_input[0, :, idx] += 1
    
    base_pred = model.predict(base_input)[1]
    increased_pred = model.predict(increased_input)[1]
    
    coefficient = (increased_pred - base_pred)[0][0]
    geo_coefficients[geo] = coefficient

# Save results
results_df = pd.DataFrame({
    'Feature': list(spend_coefficients.keys()) + list(geo_coefficients.keys()),
    'Coefficient': list(spend_coefficients.values()) + list(geo_coefficients.values())
})
results_df.to_csv('model_coefficients.csv', index=False)

print("\nModel Summary:")
model.summary()

print("\nResults saved to 'model_coefficients.csv'")