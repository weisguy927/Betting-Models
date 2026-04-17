import numpy as np
from scipy.special import expit  # Sigmoid function

def win_probability(predicted_margin):
    """
    Calculates the probability of Team 1 winning based on the predicted margin.
    
    Uses a logistic function to transform the margin into a probability.
    """
    # Logistic transformation: P(win) = 1 / (1 + exp(-k * margin))
    k = 0.2  # Scaling factor to adjust steepness of probability curve
    return expit(k * predicted_margin)

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and feature names
margin_model = joblib.load("/Users/jjweiser/Documents/margin_model.pkl")
total_model = joblib.load("/Users/jjweiser/Documents/total_model.pkl")
feature_names = joblib.load("/Users/jjweiser/Documents/model_features.pkl")

# Load dataset for team stats
file_path = "/Users/jjweiser/Documents/final_merged_dataset.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns to avoid errors
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

# Extract unique team names for dropdown menu
team_names = sorted(set(df['home_team_y']).union(set(df['away_team_y'])))

# Streamlit UI
st.title("March Madness Game Predictor")

# User selects teams from dropdown
team1 = st.selectbox("Select Team 1", team_names)
team2 = st.selectbox("Select Team 2", team_names)
seed1 = st.number_input("Enter Team 1 Seed", min_value=1, max_value=16, step=1)
seed2 = st.number_input("Enter Team 2 Seed", min_value=1, max_value=16, step=1)

if st.button("Predict Winner"):
    # Ensure accurate seeding assignments
    if seed1 < seed2:
        higher_seed_team, lower_seed_team = team1, team2
        higher_seed_value, lower_seed_value = seed1, seed2
    else:
        higher_seed_team, lower_seed_team = team2, team1
        higher_seed_value, lower_seed_value = seed2, seed1

    # Find team stats from dataset
    team1_stats = df_numeric[df['home_team_y'] == team1].mean()
    team2_stats = df_numeric[df['away_team_y'] == team2].mean()

    if team1_stats.isna().all() or team2_stats.isna().all():
        st.write("Error: No numeric stats found for one or both teams.")
    else:
        # Construct feature vector
        input_data = np.zeros(len(feature_names))
        feature_dict = dict(zip(feature_names, input_data))

        # Set team-related stats in feature_dict
        for feature in feature_names:
            if feature in team1_stats.index and not pd.isna(team1_stats[feature]):
                feature_dict[feature] = team1_stats[feature]
            if feature in team2_stats.index and not pd.isna(team2_stats[feature]):
                feature_dict[feature] = team2_stats[feature]

        # Set seed-related features accurately
        feature_dict['higher_seed'] = higher_seed_value
        feature_dict['lower_seed'] = lower_seed_value

        # Convert feature dictionary to array
        input_array = np.array([feature_dict[feature] for feature in feature_names]).reshape(1, -1)

        # Predict margin and total score
        predicted_margin = margin_model.predict(input_array)[0]
        predicted_total = total_model.predict(input_array)[0]

        # Strengthen upset correction while allowing likely upsets
        seed_diff = lower_seed_value - higher_seed_value
        upset_penalty = seed_diff * 1.5  # Stronger penalty for upsets
        adjusted_margin = predicted_margin - upset_penalty

        # Allow most common upset matchups: 5 vs 12, 6 vs 13, 7 vs 10, 8 vs 9
        likely_upset_matchups = [(5, 12), (6, 13), (7, 10), (8, 9)]
        if (higher_seed_value, lower_seed_value) in likely_upset_matchups:
            adjusted_margin += 3  # Slight boost for realistic upsets

        # Ensure realistic upsets: If margin is still close, let the higher seed win
        if adjusted_margin < 2.5:
            winner = higher_seed_team
            win_prob = win_probability(abs(adjusted_margin))  # Assign 100% probability to the higher seed in very close matchups
        else:
            win_prob = win_probability(adjusted_margin)
            winner = team1 if adjusted_margin > 0 else team2

        # Calculate predicted scores
        team1_score = (predicted_total + abs(adjusted_margin)) / 2
        team2_score = (predicted_total - abs(adjusted_margin)) / 2

        # Calculate probability of predicted spread and total score
        spread_prob = 100 * (1 / (1 + np.exp(-abs(adjusted_margin) / 10)))
        total_prob = 100 * (1 / (1 + np.exp(-abs(predicted_total - 144.78) / 15)))

        # Display results
        st.write(f"**Predicted Score:** {team1} {team1_score:.1f} - {team2} {team2_score:.1f}")
        st.write(f"**Predicted Winner:** {winner} (Win Probability: {win_prob:.1%})")
        st.write(f"**Predicted Spread:** {abs(adjusted_margin):.2f} (Probability: {spread_prob:.1f}%)")
        st.write(f"**Predicted Total:** {predicted_total:.2f} (Probability: {total_prob:.1f}%)")
