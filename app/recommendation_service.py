import glob
import joblib
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import os


class RecommendationService:
    def __init__(self):
        """Initialize the recommendation service by loading all model artifacts"""
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.info_dir = os.path.join(os.path.dirname(__file__), '..', 'model_info')
        self.encoder_dir = os.path.join(os.path.dirname(__file__), '..', 'encoder')
        self.model = None
        self.encoder = None
        self.room_lookup = None
        self.user_preferences = None
        self.feature_info = None

        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Load all saved model artifacts"""
        try:
            # Find latest model with timestamp
            model_files = glob.glob(os.path.join(self.models_dir, 'model_*.pkl'))
            if not model_files:
                raise FileNotFoundError("No versioned model files found in models directory.")
            # Sort by timestamp in filename
            latest_model_file = sorted(model_files)[-1]
            self.model = joblib.load(latest_model_file)
            print(f"✅ Loaded latest model: {os.path.basename(latest_model_file)}")

            # Load the encoder
            encoder_path = os.path.join(self.encoder_dir, 'encoder.pkl')
            self.encoder = joblib.load(encoder_path)

            # Load room lookup
            room_lookup_path = os.path.join(self.info_dir, 'room_lookup.json')
            with open(room_lookup_path, 'r') as f:
                self.room_lookup = json.load(f)

            # Load user preferences
            user_pref_path = os.path.join(self.info_dir, 'user_preferences.json')
            with open(user_pref_path, 'r') as f:
                self.user_preferences = json.load(f)

            # Load model info
            info_path = os.path.join(self.info_dir, 'model_info.json')
            with open(info_path, 'r') as f:
                self.feature_info = json.load(f)

        except Exception as e:
            print(f"❌ Error loading model artifacts: {str(e)}")
            raise

    def get_candidate_slots(self, user_id: int, purpose: str, attendees: int,
                            target_date: datetime, target_hours: List[int]) -> pd.DataFrame:
        """Generate candidate slots for a booking request"""

        candidates = []
        user_preferred_rooms = self.user_preferences.get(str(user_id), [])

        for hour in target_hours:
            for room_id in self.room_lookup.keys():
                start_time = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)

                candidate = {
                    'user_id': user_id,
                    'purpose': purpose,
                    'room_type': self.room_lookup[room_id]['room_type'],
                    'has_projector': self.room_lookup[room_id]['has_projector'],
                    'has_whiteboard': self.room_lookup[room_id]['has_whiteboard'],
                    'attendees': attendees,
                    'room_capacity': self.room_lookup[room_id]['room_capacity'],
                    'hour_of_day': hour,
                    'day_of_week': start_time.weekday(),
                    'is_weekend': 1 if start_time.weekday() >= 5 else 0,
                    'is_preferred_room': 1 if room_id in user_preferred_rooms else 0,
                    'capacity_utilization': attendees / self.room_lookup[room_id]['room_capacity'],
                    'season': ((start_time.month % 12) // 3) + 1,
                    'room_id': room_id,
                    'start_time': start_time
                }

                candidates.append(candidate)

        return pd.DataFrame(candidates)

    def recommend_slots(self, user_id: int, purpose: str, attendees: int,
                        target_date: datetime, target_hours: List[int],
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """Get slot recommendations for a booking request"""

        # Generate candidates
        candidates = self.get_candidate_slots(user_id, purpose, attendees,
                                              target_date, target_hours)

        # Prepare features
        X_candidate = candidates[self.feature_info['features']]

        # Split categorical and numerical features
        cat_features = self.feature_info['categorical_features']
        X_cat = X_candidate[cat_features]
        X_num = X_candidate.drop(columns=cat_features)

        # Encode categorical features
        X_encoded_cat = pd.DataFrame(
            self.encoder.transform(X_cat).toarray(),
            columns=self.encoder.get_feature_names_out(cat_features)
        )

        # Combine features
        X_final = pd.concat([X_encoded_cat, X_num.reset_index(drop=True)], axis=1)

        # Get predictions
        success_probabilities = self.model.predict_proba(X_final)[:, 1]
        candidates['success_probability'] = success_probabilities

        # Sort and get top recommendations
        top_recommendations = candidates.sort_values('success_probability', ascending=False).head(top_k)

        # Convert to list of dictionaries for API response
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                'room_id': row['room_id'],
                'start_time': row['start_time'].isoformat(),
                'success_probability': float(row['success_probability']),
                'room_type': row['room_type'],
                'room_capacity': int(row['room_capacity']),
                'has_projector': bool(row['has_projector']),
                'has_whiteboard': bool(row['has_whiteboard']),
                'capacity_utilization': float(row['capacity_utilization'])
            })

        return recommendations

    def get_room_info(self, room_id: str = None) -> Dict[str, Any]:
        """Get information about rooms"""
        if room_id:
            return self.room_lookup.get(room_id, {})
        return self.room_lookup

    def get_user_preferences(self, user_id: int) -> List[str]:
        """Get user's preferred rooms"""
        return self.user_preferences.get(str(user_id), [])