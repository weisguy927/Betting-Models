import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load trained Random Forest models
home_model = joblib.load("random_forest_home_model.pkl")
away_model = joblib.load("random_forest_away_model.pkl")

# Load processed dataset
data_path = "processed_ncaa_data.csv"
df = pd.read_csv(data_path)

def get_team_stats(team_name, df):
    """Retrieve team statistics for prediction."""
    team_stats = df[df['home_team'] == team_name].iloc[0]  # Get first occurrence
    return team_stats[[
        "home_SOS", "home_Ftr", "home_TS%", "home_eFG%", "home_TOV%", "home_ORB%", 
        "home_AdjOE", "home_AdjDE", "home_AdjT"
    ]].values

def predict_matchup(home_team, away_team, df, home_model, away_model, lambda_factor):
    """Predict matchup score and win probabilities."""
    home_features = get_team_stats(home_team, df)
    away_features = get_team_stats(away_team, df)
    
    # Combine features for prediction
    feature_columns = [
        "home_SOS", "home_Ftr", "home_TS%", "home_eFG%", "home_TOV%", "home_ORB%", "home_AdjOE", "home_AdjDE", "home_AdjT",
        "away_SOS", "away_Ftr", "away_TS%", "away_eFG%", "away_TOV%", "away_ORB%", "away_AdjOE", "away_AdjDE", "away_AdjT"
    ]
    matchup_features = np.concatenate((home_features, away_features)).reshape(1, -1)
    matchup_df = pd.DataFrame(matchup_features, columns=feature_columns)  # Convert to DataFrame
    
    # Predict scores
    predicted_home_score = home_model.predict(matchup_df)[0]
    predicted_away_score = away_model.predict(matchup_df)[0]
    
    # Compute score difference
    score_diff = predicted_home_score - predicted_away_score
    
    # Calculate win probability using logistic function
    home_win_prob = 1 / (1 + np.exp(-lambda_factor * score_diff))
    away_win_prob = 1 - home_win_prob
    
    # Determine winner
    winner = home_team if predicted_home_score > predicted_away_score else away_team
    
    return predicted_home_score, predicted_away_score, home_win_prob, away_win_prob, winner

# Streamlit UI
st.title("NCAA Basketball Matchup Predictor")

home_team = st.selectbox("Select Home Team", df["home_team"].unique())
away_team = st.selectbox("Select Away Team", df["away_team"].unique())

# Add slider to adjust lambda factor
lambda_factor = st.slider("Adjust Confidence Factor (Lambda)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

if st.button("Predict Outcome"):
    home_score, away_score, home_win_prob, away_win_prob, winner = predict_matchup(home_team, away_team, df, home_model, away_model, lambda_factor)
    
    st.subheader("Predicted Scores")
    st.write(f"**{home_team}:** {round(home_score, 1)}")
    st.write(f"**{away_team}:** {round(away_score, 1)}")
    
    st.subheader("Win Probabilities")
    st.write(f"🏠 **{home_team} Win Probability:** {round(home_win_prob * 100, 1)}%")
    st.write(f"✈️ **{away_team} Win Probability:** {round(away_win_prob * 100, 1)}%")
    
    st.subheader("Predicted Winner")
    st.write(f"🏆 **{winner}**")
