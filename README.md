# Booking Recommendation System

## Important Setup Instructions

This project expects the following directories to exist before running:

- `data/`
- `models/`
- `model_info/`
- `encoder/`
- `logs/`

### Why?

Git does **not** track empty folders, so these directories might not be present after cloning.

---

## How to create the required folders

To ensure the project runs smoothly, please create these folders manually **and add a `.keep` file** inside each, like this:

```bash
mkdir -p data models model_info encoder logs

# Create placeholder files to keep folders tracked by Git
touch data/.keep models/.keep model_info/.keep encoder/.keep logs/.keep
