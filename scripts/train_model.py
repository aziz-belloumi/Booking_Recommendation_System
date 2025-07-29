import pickle

import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
from datetime import datetime

# generating a time stamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

# models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
# encoder directory
encoder_dir = os.path.join(os.path.dirname(__file__), '..', 'encoder')

# Load the dataset
df = pd.read_csv("../data/dataset.csv", parse_dates=["start_time", "end_time"])

# Create binary target label
df['target'] = ((df['overloaded'] == 0) &
                (df['conflict_flag'] == 0) &
                (df['anomaly_flag'] == 0)).astype(int)

# Features used for training
features = [
    'user_id', 'purpose', 'room_type',
    'has_projector', 'has_whiteboard', 'attendees', 'room_capacity',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_preferred_room',
    'capacity_utilization', 'season'
]

X_raw = df[features]
y = df['target']

# Split categorical and numerical features
X_cat = X_raw[['user_id', 'purpose', 'room_type']]
X_num = X_raw.drop(columns=X_cat.columns)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded_cat = pd.DataFrame(
    encoder.fit_transform(X_cat).toarray(),
    columns=encoder.get_feature_names_out(X_cat.columns)
)

# Combine encoded categorical and numerical features
X_encoded = pd.concat([X_encoded_cat, X_num.reset_index(drop=True)], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


json_dir = os.path.join(os.path.dirname(__file__), '..', 'model_info')

# Create room lookup dictionary
room_info = df[['room_id', 'room_capacity', 'room_type', 'has_projector', 'has_whiteboard']].drop_duplicates()
room_lookup = room_info.set_index('room_id').to_dict('index')

# Convert boolean values to native Python types for JSON
room_lookup_json = {
    room_id: {
        'room_capacity': int(info['room_capacity']),
        'room_type': str(info['room_type']),
        'has_projector': bool(info['has_projector']),
        'has_whiteboard': bool(info['has_whiteboard'])
    }
    for room_id, info in room_lookup.items()
}


# Save room_lookup.json
with open(os.path.join(json_dir, 'room_lookup.json'), 'w') as f:
    json.dump(room_lookup_json, f, indent=2)

# Save model feature info
feature_info = {
    'features': features,
    'categorical_features': ['user_id', 'purpose', 'room_type'],
    'model_version': '1.0',
    'trained_date': datetime.now().isoformat()
}
with open(os.path.join(json_dir, 'model_info.json'), 'w') as f:
    json.dump(feature_info, f, indent=2)



# Create a user preferences dictionary
user_preferences = {}
for user_id in df['user_id'].unique():
    preferred_rooms = df[df['user_id'] == user_id]['room_id'].unique().tolist()
    user_preferences[str(user_id)] = preferred_rooms

# Save user_preferences.json
with open(os.path.join(json_dir, 'user_preferences.json'), 'w') as f:
    json.dump(user_preferences, f, indent=2)


# Save model with timestamp
model_path = os.path.join(models_dir, f'model_{timestamp}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# save the encoder
joblib.dump(encoder, os.path.join(encoder_dir, 'encoder.pkl'))


# logs directory
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

log_file_path = os.path.join(logs_dir, f"training_{timestamp}.log")
# Write to log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Model Training Log - {timestamp}\n")
    log_file.write(f"Total Records: {len(df)}\n")
    log_file.write(f"Success Rate (target mean): {df['target'].mean() * 100:.2f}%\n")
    log_file.write(f"Accuracy Score: {accuracy * 100:.2f}%\n\n")
    log_file.write("Classification Report:\n")
    log_file.write(report)
