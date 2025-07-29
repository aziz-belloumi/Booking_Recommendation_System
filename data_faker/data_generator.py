import pandas as pd
import random
from faker import Faker
from datetime import timedelta
import numpy as np


# Set seeds for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Constants
NUM_ENTRIES = 300_000
NUM_USERS = 100
NUM_ROOMS = 25

# Generate rooms with random capacities and features
rooms = [{
    "room_id": f"R{i + 1}",
    "capacity": random.randint(5, 30),
    "has_projector": random.choice([True, False]),
    "has_whiteboard": random.choice([True, False]),
    "room_type": random.choice(["meeting", "training", "interview", "flex"])
} for i in range(NUM_ROOMS)]

# Booking purposes with required room features
purposes = {
    "Team meeting": {"room_type": ["meeting", "flex"], "min_capacity": 8},
    "Project presentation": {"has_projector": True, "room_type": ["training", "meeting"]},
    "Interview": {"room_type": ["interview"], "max_noise": True},
    "Training session": {"has_whiteboard": True, "min_capacity": 10},
    "Client meeting": {"room_type": ["meeting"], "min_capacity": 5},
    "Workshop": {"has_whiteboard": True, "room_type": ["training"]},
    "Conference call": {"room_type": ["flex", "meeting"], "min_capacity": 3},
    "Brainstorming": {"has_whiteboard": True, "room_type": ["meeting", "flex"]},
    "Demo": {"has_projector": True, "room_type": ["meeting", "training"]},
    "One-on-one": {"room_type": ["interview", "flex"], "max_capacity": 5}
}

# User-room preferences (each user prefers rooms matching their needs)
user_preferences = {}
for user_id in range(1, NUM_USERS + 1):
    preferred_rooms = []
    for room in rooms:
        # Logic for user preferences based on user_id patterns
        if (user_id % 5 == 0 and room["has_projector"]) or \
                (user_id % 3 == 0 and room["room_type"] == "interview") or \
                (user_id % 7 == 0 and room["has_whiteboard"]) or \
                (random.random() < 0.3):  # 30% chance for random preference
            preferred_rooms.append(room["room_id"])

    # Ensure each user has at least 2 preferred rooms
    if len(preferred_rooms) < 2:
        preferred_rooms.extend([room["room_id"] for room in random.sample(rooms, 2)])
        preferred_rooms = list(set(preferred_rooms))

    user_preferences[user_id] = preferred_rooms[:5]  # Keep the top 5 preferred rooms


def check_room_compatibility(room, purpose_reqs):
    """Check if a room meets purpose requirements"""
    for key, value in purpose_reqs.items():
        if key == "min_capacity":
            if room["capacity"] < value:
                return False
        elif key == "max_capacity":
            if room["capacity"] > value:
                return False
        elif key == "room_type":
            if isinstance(value, list):
                if room["room_type"] not in value:
                    return False
            else:
                if room["room_type"] != value:
                    return False
        elif key in ["has_projector", "has_whiteboard"]:
            if room.get(key) != value:
                return False
        # Skip other keys like "max_noise" for now
    return True


# Generate data in batches to avoid memory issues
data = []
batch_size = 10000

print("Generating booking data...")
for batch in range(0, NUM_ENTRIES, batch_size):
    batch_data = []
    current_batch_size = min(batch_size, NUM_ENTRIES - batch)

    for i in range(current_batch_size):
        if (batch + i) % 10000 == 0:
            print(f"Progress: {batch + i}/{NUM_ENTRIES} ({((batch + i) / NUM_ENTRIES) * 100:.1f}%)")

        user_id = random.randint(1, NUM_USERS)
        purpose = random.choice(list(purposes.keys()))
        purpose_reqs = purposes[purpose]

        # Filter rooms that match purpose requirements
        compatible_rooms = [
            r for r in rooms
            if check_room_compatibility(r, purpose_reqs)
        ]

        # Fallback to all rooms if no matches
        if not compatible_rooms:
            compatible_rooms = rooms

        # 70% chance to use a user's preferred AND compatible room
        selected_room = None
        if random.random() < 0.7:
            preferred_compatible = [
                r for r in compatible_rooms
                if r["room_id"] in user_preferences[user_id]
            ]
            if preferred_compatible:
                selected_room = random.choice(preferred_compatible)

        if selected_room is None:
            selected_room = random.choice(compatible_rooms)

        # Generate time with a 10% chance of anomaly (after hours)
        start_time = fake.date_time_between(start_date='-90d', end_date='now')
        if random.random() < 0.1:  # Anomaly: 10% odd-hour bookings
            start_time = start_time.replace(hour=random.choice([5, 6, 20, 21, 22]))

        duration_minutes = random.choice([30, 60, 90, 120])
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Generate attendees (sometimes exceeding capacity)
        base_attendees = random.randint(1, selected_room["capacity"])
        # 15% chance of overbooking
        if random.random() < 0.15:
            attendees = base_attendees + random.randint(1, 5)
        else:
            attendees = base_attendees

        # Generate description that matches purpose
        description_templates = {
            "Team meeting": f"Team sync on {fake.bs()}",
            "Project presentation": f"Presentation for {fake.job()} stakeholders",
            "Interview": f"Interview with {fake.name()} for {fake.job()} role",
            "Training session": f"Training on {fake.catch_phrase()}",
            "Client meeting": f"Client meeting with {fake.company()}",
            "Workshop": f"Workshop on {fake.bs()}",
            "Conference call": f"Conference call with {fake.company()}",
            "Brainstorming": f"Brainstorming session for {fake.catch_phrase()}",
            "Demo": f"Demo of {fake.catch_phrase()}",
            "One-on-one": f"One-on-one with {fake.name()}"
        }
        description = description_templates.get(purpose, f"{purpose}: {fake.sentence()}")

        batch_data.append({
            "user_id": user_id,
            "room_id": selected_room["room_id"],
            "room_capacity": selected_room["capacity"],
            "room_type": selected_room["room_type"],
            "has_projector": selected_room["has_projector"],
            "has_whiteboard": selected_room["has_whiteboard"],
            "purpose": purpose,
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration_minutes,
            "attendees": attendees,
            "overloaded": 1 if attendees > selected_room["capacity"] else 0,
            "description": description,
            "conflict_flag": 1 if random.random() < 0.1 else 0,
            "anomaly_flag": 1 if (start_time.hour < 7 or start_time.hour > 19) else 0,
            "is_preferred_room": 1 if selected_room["room_id"] in user_preferences[user_id] else 0,
            "is_purpose_compatible": 1,  # Always 1 due to our filtering
            "day_of_week": start_time.weekday(),
            "hour_of_day": start_time.hour,
            "month": start_time.month,
            "is_weekend": 1 if start_time.weekday() >= 5 else 0
        })

    data.extend(batch_data)

# Create DataFrame
print("Creating DataFrame...")
df = pd.DataFrame(data)

# Add some additional features for ML
df['capacity_utilization'] = df['attendees'] / df['room_capacity']
df['is_peak_hour'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
df['season'] = ((df['month'] % 12) // 3) + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

# Save to CSV
print("Saving to CSV...")
df.to_csv("../data/dataset.csv", index=False)

print(f"✅ Successfully generated {len(df)} bookings")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Display some statistics
print("\n📊 Dataset Statistics:")
print(f"- Users: {df['user_id'].nunique()}")
print(f"- Rooms: {df['room_id'].nunique()}")
print(f"- Purposes: {df['purpose'].nunique()}")
print(f"- Date range: {df['start_time'].min()} to {df['start_time'].max()}")
print(f"- Overloaded bookings: {df['overloaded'].sum()} ({df['overloaded'].mean() * 100:.1f}%)")
print(f"- Anomaly bookings: {df['anomaly_flag'].sum()} ({df['anomaly_flag'].mean() * 100:.1f}%)")
print(f"- Preferred room usage: {df['is_preferred_room'].mean() * 100:.1f}%")