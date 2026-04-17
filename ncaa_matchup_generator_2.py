import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error


EXCEL_FILE = "ncaa_schedule_compiled_26.xlsx"

SCHEDULE_SHEET = "Schedule"
NAME_MATCH_SHEET = "Name Match"
TEAM_RESULTS_SHEET = "2026_team_results"

MODEL_RANDOM_STATE = 42
FORM_WINDOWS = [3, 5, 10]
MODEL_START_DATE = pd.Timestamp("2025-12-01", tz="UTC")


def clean_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def find_column(columns, candidates):
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def load_workbook(path: str):
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names

    required = [SCHEDULE_SHEET, NAME_MATCH_SHEET, TEAM_RESULTS_SHEET]
    missing = [s for s in required if s not in sheet_names]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}. Available: {sheet_names}")

    schedule_df = pd.read_excel(path, sheet_name=SCHEDULE_SHEET)
    name_match_df = pd.read_excel(path, sheet_name=NAME_MATCH_SHEET)
    team_results_df = pd.read_excel(path, sheet_name=TEAM_RESULTS_SHEET)

    return schedule_df, name_match_df, team_results_df, sheet_names


def build_name_map(name_match_df: pd.DataFrame) -> dict:
    cols = list(name_match_df.columns)

    source_col = find_column(cols, [
        "source_name", "source", "raw_name", "original_name",
        "schedule_name", "alias", "team", "schedule team",
        "schedule_team", "name1"
    ])

    target_col = find_column(cols, [
        "matched_name", "standard_name", "canonical_name",
        "team_name", "standardized_name", "matched team",
        "matched_team", "name2"
    ])

    if source_col is None or target_col is None:
        raise ValueError(
            f"Could not identify name map columns. Found: {cols}"
        )

    mapping = {}
    for _, row in name_match_df.iterrows():
        src = clean_name(row[source_col])
        tgt = clean_name(row[target_col])
        if src and tgt:
            mapping[src] = tgt

    if not mapping:
        raise ValueError("No usable team name mappings were created.")

    return mapping


def standardize_team(name: str, mapping: dict) -> str:
    cleaned = clean_name(name)
    return mapping.get(cleaned, cleaned)


def prepare_games(schedule_df: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    cols = list(schedule_df.columns)

    date_col = find_column(cols, ["date", "game_date", "start_date", "startdate"])
    team1_col = find_column(cols, ["home_team", "home team", "home", "team_home"])
    team2_col = find_column(cols, ["away_team", "away team", "away", "team_away"])
    score1_col = find_column(cols, ["home_score", "home score", "score_home"])
    score2_col = find_column(cols, ["away_score", "away score", "score_away"])

    missing = [
        name for name, col in {
            "date": date_col,
            "team_1": team1_col,
            "team_2": team2_col,
            "team_1_score": score1_col,
            "team_2_score": score2_col,
        }.items() if col is None
    ]
    if missing:
        raise ValueError(f"Missing required schedule columns: {missing}. Available: {cols}")

    games = schedule_df.copy()
    games["game_date"] = pd.to_datetime(games[date_col], errors="coerce", utc=True)
    games["team_1"] = games[team1_col].apply(lambda x: standardize_team(x, name_map))
    games["team_2"] = games[team2_col].apply(lambda x: standardize_team(x, name_map))
    games["team_1_score"] = pd.to_numeric(games[score1_col], errors="coerce")
    games["team_2_score"] = pd.to_numeric(games[score2_col], errors="coerce")

    games["is_completed"] = games["team_1_score"].notna() & games["team_2_score"].notna()
    games["team_1_win"] = np.where(
        games["is_completed"],
        (games["team_1_score"] > games["team_2_score"]).astype(int),
        np.nan
    )
    games["actual_total"] = games["team_1_score"] + games["team_2_score"]

    games = games[games["game_date"] >= MODEL_START_DATE].copy()
    return games.sort_values("game_date").reset_index(drop=True)


def prepare_team_results(team_results_df: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    df = team_results_df.copy()

    required_cols = ["team", "adjoe", "adjde", "barthag", "adjt", "sos", "WAB", "Conf Win%"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required team results columns: {missing}")

    df["team_std"] = df["team"].apply(lambda x: standardize_team(x, name_map))

    numeric_cols = ["adjoe", "adjde", "barthag", "adjt", "sos", "WAB", "Conf Win%"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["team_std"] + numeric_cols
    df = df[keep_cols].drop_duplicates(subset=["team_std"]).copy()
    return df


def build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    completed = games[games["is_completed"]].copy()

    team1_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_1"],
        "opponent": completed["team_2"],
        "points_for": completed["team_1_score"],
        "points_against": completed["team_2_score"],
        "win": (completed["team_1_score"] > completed["team_2_score"]).astype(int),
        "game_total": completed["actual_total"],
    })

    team2_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_2"],
        "opponent": completed["team_1"],
        "points_for": completed["team_2_score"],
        "points_against": completed["team_1_score"],
        "win": (completed["team_2_score"] > completed["team_1_score"]).astype(int),
        "game_total": completed["actual_total"],
    })

    team_log = pd.concat([team1_rows, team2_rows], ignore_index=True)
    team_log["scoring_margin"] = team_log["points_for"] - team_log["points_against"]
    team_log = team_log.sort_values(["team", "game_date"]).reset_index(drop=True)

    for window in FORM_WINDOWS:
        team_log[f"rolling_win_pct_{window}"] = (
            team_log.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        team_log[f"rolling_points_for_{window}"] = (
            team_log.groupby("team")["points_for"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        team_log[f"rolling_points_against_{window}"] = (
            team_log.groupby("team")["points_against"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        team_log[f"rolling_scoring_margin_{window}"] = (
            team_log.groupby("team")["scoring_margin"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        team_log[f"rolling_game_total_{window}"] = (
            team_log.groupby("team")["game_total"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    team_log["games_played_prior"] = team_log.groupby("team").cumcount()
    return team_log


def get_recent_form_before_date(team_log: pd.DataFrame, team: str, game_date: pd.Timestamp) -> dict:
    rows = team_log[(team_log["team"] == team) & (team_log["game_date"] < game_date)]

    if rows.empty:
        default = {"games_played_prior": 0}
        for window in FORM_WINDOWS:
            default[f"recent_win_pct_{window}"] = 0.5
            default[f"recent_points_for_{window}"] = 70.0
            default[f"recent_points_against_{window}"] = 70.0
            default[f"recent_scoring_margin_{window}"] = 0.0
            default[f"recent_game_total_{window}"] = 140.0
        return default

    last = rows.sort_values("game_date").iloc[-1]
    result = {
        "games_played_prior": int(last["games_played_prior"]) if pd.notna(last["games_played_prior"]) else 0
    }

    for window in FORM_WINDOWS:
        result[f"recent_win_pct_{window}"] = float(last[f"rolling_win_pct_{window}"]) if pd.notna(last[f"rolling_win_pct_{window}"]) else 0.5
        result[f"recent_points_for_{window}"] = float(last[f"rolling_points_for_{window}"]) if pd.notna(last[f"rolling_points_for_{window}"]) else 70.0
        result[f"recent_points_against_{window}"] = float(last[f"rolling_points_against_{window}"]) if pd.notna(last[f"rolling_points_against_{window}"]) else 70.0
        result[f"recent_scoring_margin_{window}"] = float(last[f"rolling_scoring_margin_{window}"]) if pd.notna(last[f"rolling_scoring_margin_{window}"]) else 0.0
        result[f"recent_game_total_{window}"] = float(last[f"rolling_game_total_{window}"]) if pd.notna(last[f"rolling_game_total_{window}"]) else 140.0

    return result


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["adjoe_diff"] = df["team_1_adjoe"] - df["team_2_adjoe"]
    df["adjde_diff"] = df["team_1_adjde"] - df["team_2_adjde"]
    df["barthag_diff"] = df["team_1_barthag"] - df["team_2_barthag"]
    df["adjt_diff"] = df["team_1_adjt"] - df["team_2_adjt"]
    df["sos_diff"] = df["team_1_sos"] - df["team_2_sos"]
    df["WAB_diff"] = df["team_1_WAB"] - df["team_2_WAB"]
    df["conf_win_pct_diff"] = df["team_1_conf_win_pct"] - df["team_2_conf_win_pct"]
    df["games_played_prior_diff"] = df["team_1_games_played_prior"] - df["team_2_games_played_prior"]

    df["combined_adjoe"] = df["team_1_adjoe"] + df["team_2_adjoe"]
    df["combined_adjde"] = df["team_1_adjde"] + df["team_2_adjde"]
    df["combined_adjt"] = df["team_1_adjt"] + df["team_2_adjt"]
    df["avg_adjt"] = (df["team_1_adjt"] + df["team_2_adjt"]) / 2.0
    df["tempo_gap"] = (df["team_1_adjt"] - df["team_2_adjt"]).abs()

    for window in FORM_WINDOWS:
        df[f"recent_win_pct_diff_{window}"] = df[f"team_1_recent_win_pct_{window}"] - df[f"team_2_recent_win_pct_{window}"]
        df[f"recent_points_for_diff_{window}"] = df[f"team_1_recent_points_for_{window}"] - df[f"team_2_recent_points_for_{window}"]
        df[f"recent_points_against_diff_{window}"] = df[f"team_1_recent_points_against_{window}"] - df[f"team_2_recent_points_against_{window}"]
        df[f"recent_scoring_margin_diff_{window}"] = df[f"team_1_recent_scoring_margin_{window}"] - df[f"team_2_recent_scoring_margin_{window}"]
        df[f"recent_game_total_avg_{window}"] = (
            df[f"team_1_recent_game_total_{window}"] + df[f"team_2_recent_game_total_{window}"]
        ) / 2.0

    return df


def build_training_dataset(games: pd.DataFrame, team_results: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    model_df = games.copy()

    team1_stats = team_results.rename(columns={
        "team_std": "team_1",
        "adjoe": "team_1_adjoe",
        "adjde": "team_1_adjde",
        "barthag": "team_1_barthag",
        "adjt": "team_1_adjt",
        "sos": "team_1_sos",
        "WAB": "team_1_WAB",
        "Conf Win%": "team_1_conf_win_pct",
    })
    team2_stats = team_results.rename(columns={
        "team_std": "team_2",
        "adjoe": "team_2_adjoe",
        "adjde": "team_2_adjde",
        "barthag": "team_2_barthag",
        "adjt": "team_2_adjt",
        "sos": "team_2_sos",
        "WAB": "team_2_WAB",
        "Conf Win%": "team_2_conf_win_pct",
    })

    model_df = model_df.merge(team1_stats, on="team_1", how="left")
    model_df = model_df.merge(team2_stats, on="team_2", how="left")

    team_1_form_data = []
    team_2_form_data = []

    for _, row in model_df.iterrows():
        game_date = row["game_date"]
        team_1_form_data.append(get_recent_form_before_date(team_log, row["team_1"], game_date))
        team_2_form_data.append(get_recent_form_before_date(team_log, row["team_2"], game_date))

    team_1_form_df = pd.DataFrame(team_1_form_data).add_prefix("team_1_")
    team_2_form_df = pd.DataFrame(team_2_form_data).add_prefix("team_2_")

    model_df = pd.concat([model_df.reset_index(drop=True), team_1_form_df, team_2_form_df], axis=1)
    model_df = add_engineered_features(model_df)
    return model_df


def build_feature_columns():
    feature_cols = [
        "team_1_adjoe", "team_1_adjde", "team_1_barthag", "team_1_adjt",
        "team_1_sos", "team_1_WAB", "team_1_conf_win_pct",
        "team_2_adjoe", "team_2_adjde", "team_2_barthag", "team_2_adjt",
        "team_2_sos", "team_2_WAB", "team_2_conf_win_pct",
        "adjoe_diff", "adjde_diff", "barthag_diff", "adjt_diff",
        "sos_diff", "WAB_diff", "conf_win_pct_diff",
        "team_1_games_played_prior", "team_2_games_played_prior",
        "games_played_prior_diff",
        "combined_adjoe", "combined_adjde", "combined_adjt",
        "avg_adjt", "tempo_gap",
    ]

    for side in ["team_1", "team_2"]:
        for window in FORM_WINDOWS:
            feature_cols.extend([
                f"{side}_recent_win_pct_{window}",
                f"{side}_recent_points_for_{window}",
                f"{side}_recent_points_against_{window}",
                f"{side}_recent_scoring_margin_{window}",
                f"{side}_recent_game_total_{window}",
            ])

    for window in FORM_WINDOWS:
        feature_cols.extend([
            f"recent_win_pct_diff_{window}",
            f"recent_points_for_diff_{window}",
            f"recent_points_against_diff_{window}",
            f"recent_scoring_margin_diff_{window}",
            f"recent_game_total_avg_{window}",
        ])

    return feature_cols


def make_symmetric_training_dataset(completed_df: pd.DataFrame) -> pd.DataFrame:
    original = completed_df.copy()
    swapped = completed_df.copy()

    swap_pairs = [
        ("team_1", "team_2"),
        ("team_1_score", "team_2_score"),
        ("team_1_adjoe", "team_2_adjoe"),
        ("team_1_adjde", "team_2_adjde"),
        ("team_1_barthag", "team_2_barthag"),
        ("team_1_adjt", "team_2_adjt"),
        ("team_1_sos", "team_2_sos"),
        ("team_1_WAB", "team_2_WAB"),
        ("team_1_conf_win_pct", "team_2_conf_win_pct"),
        ("team_1_games_played_prior", "team_2_games_played_prior"),
    ]

    for window in FORM_WINDOWS:
        swap_pairs.extend([
            (f"team_1_recent_win_pct_{window}", f"team_2_recent_win_pct_{window}"),
            (f"team_1_recent_points_for_{window}", f"team_2_recent_points_for_{window}"),
            (f"team_1_recent_points_against_{window}", f"team_2_recent_points_against_{window}"),
            (f"team_1_recent_scoring_margin_{window}", f"team_2_recent_scoring_margin_{window}"),
            (f"team_1_recent_game_total_{window}", f"team_2_recent_game_total_{window}"),
        ])

    for left_col, right_col in swap_pairs:
        if left_col in swapped.columns and right_col in swapped.columns:
            swapped[left_col], swapped[right_col] = original[right_col], original[left_col]

    swapped["team_1_win"] = 1 - original["team_1_win"].astype(int)
    swapped = add_engineered_features(swapped)

    symmetric = pd.concat([original, swapped], ignore_index=True)
    return symmetric.sort_values(["game_date", "team_1", "team_2"]).reset_index(drop=True)


def time_based_split(train_df: pd.DataFrame):
    unique_dates = sorted(train_df["game_date"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough unique game dates for time-based split.")

    split_idx = int(len(unique_dates) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = unique_dates[split_idx]

    train_part = train_df[train_df["game_date"] < split_date].copy()
    test_part = train_df[train_df["game_date"] >= split_date].copy()

    if train_part.empty or test_part.empty:
        raise ValueError(
            f"Time-based split failed. Split date: {split_date}, "
            f"train rows: {len(train_part)}, test rows: {len(test_part)}"
        )

    return train_part, test_part, split_date


def train_models(model_df: pd.DataFrame):
    feature_cols = build_feature_columns()

    completed_df = model_df[model_df["is_completed"]].copy()
    completed_df = completed_df.dropna(subset=feature_cols + ["team_1_win", "team_1_score", "team_2_score", "game_date"])

    if len(completed_df) < 50:
        raise ValueError(f"Not enough completed games to train. Found {len(completed_df)} rows.")

    symmetric_df = make_symmetric_training_dataset(completed_df)
    train_part, test_part, split_date = time_based_split(symmetric_df)

    X_train = train_part[feature_cols]
    X_test = test_part[feature_cols]

    y_win_train = train_part["team_1_win"].astype(int)
    y_win_test = test_part["team_1_win"].astype(int)

    y_score1_train = train_part["team_1_score"]
    y_score1_test = test_part["team_1_score"]

    y_score2_train = train_part["team_2_score"]
    y_score2_test = test_part["team_2_score"]

    win_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    score1_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=MODEL_RANDOM_STATE + 1,
        n_jobs=-1,
    )
    score2_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=MODEL_RANDOM_STATE + 2,
        n_jobs=-1,
    )

    win_model.fit(X_train, y_win_train)
    score1_model.fit(X_train, y_score1_train)
    score2_model.fit(X_train, y_score2_train)

    win_probs = win_model.predict_proba(X_test)[:, 1]
    win_preds = (win_probs >= 0.5).astype(int)
    score1_preds = score1_model.predict(X_test)
    score2_preds = score2_model.predict(X_test)

    print("\nModel evaluation")
    print("----------------")
    print(f"Completed rows used: {len(completed_df)}")
    print(f"Symmetric rows used: {len(symmetric_df)}")
    print(f"Train rows: {len(train_part)}")
    print(f"Test rows: {len(test_part)}")
    print(f"Time split date: {pd.Timestamp(split_date).date()}")

    print("\nWin model")
    print(f"Accuracy: {accuracy_score(y_win_test, win_preds):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_win_test, win_probs):.3f}")

    print("\nTeam 1 score model")
    print(f"MAE:  {mean_absolute_error(y_score1_test, score1_preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_score1_test, score1_preds)):.3f}")

    print("\nTeam 2 score model")
    print(f"MAE:  {mean_absolute_error(y_score2_test, score2_preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_score2_test, score2_preds)):.3f}")

    return {
        "win_model": win_model,
        "score1_model": score1_model,
        "score2_model": score2_model,
        "feature_cols": feature_cols,
    }


def canonicalize_input_team(team_name: str, name_map: dict, team_results: pd.DataFrame) -> str:
    cleaned = standardize_team(team_name, name_map)
    valid_teams = set(team_results["team_std"].dropna().astype(str))

    if cleaned in valid_teams:
        return cleaned

    candidates = sorted(valid_teams)
    contains = [t for t in candidates if cleaned in t or t in cleaned]

    if len(contains) == 1:
        return contains[0]

    raise ValueError(
        f"Team '{team_name}' could not be matched to a team in {TEAM_RESULTS_SHEET}. "
        f"Closest contains-matches: {contains[:10]}"
    )


def build_matchup_row(
    team_1: str,
    team_2: str,
    prediction_date: pd.Timestamp,
    team_results: pd.DataFrame,
    team_log: pd.DataFrame,
) -> pd.DataFrame:
    team1_stats = team_results[team_results["team_std"] == team_1]
    team2_stats = team_results[team_results["team_std"] == team_2]

    if team1_stats.empty:
        raise ValueError(f"No team results found for team_1: {team_1}")
    if team2_stats.empty:
        raise ValueError(f"No team results found for team_2: {team_2}")

    row = pd.DataFrame({
        "game_date": [prediction_date],
        "team_1": [team_1],
        "team_2": [team_2],
    })

    row["team_1_adjoe"] = team1_stats["adjoe"].iloc[0]
    row["team_1_adjde"] = team1_stats["adjde"].iloc[0]
    row["team_1_barthag"] = team1_stats["barthag"].iloc[0]
    row["team_1_adjt"] = team1_stats["adjt"].iloc[0]
    row["team_1_sos"] = team1_stats["sos"].iloc[0]
    row["team_1_WAB"] = team1_stats["WAB"].iloc[0]
    row["team_1_conf_win_pct"] = team1_stats["Conf Win%"].iloc[0]

    row["team_2_adjoe"] = team2_stats["adjoe"].iloc[0]
    row["team_2_adjde"] = team2_stats["adjde"].iloc[0]
    row["team_2_barthag"] = team2_stats["barthag"].iloc[0]
    row["team_2_adjt"] = team2_stats["adjt"].iloc[0]
    row["team_2_sos"] = team2_stats["sos"].iloc[0]
    row["team_2_WAB"] = team2_stats["WAB"].iloc[0]
    row["team_2_conf_win_pct"] = team2_stats["Conf Win%"].iloc[0]

    team_1_form = get_recent_form_before_date(team_log, team_1, prediction_date)
    team_2_form = get_recent_form_before_date(team_log, team_2, prediction_date)

    for key, value in team_1_form.items():
        row[f"team_1_{key}"] = value
    for key, value in team_2_form.items():
        row[f"team_2_{key}"] = value

    row = add_engineered_features(row)
    return row


def predict_matchup(
    team_1: str,
    team_2: str,
    models: dict,
    team_results: pd.DataFrame,
    team_log: pd.DataFrame,
    prediction_date: pd.Timestamp,
) -> dict:
    # first order
    row_ab = build_matchup_row(team_1, team_2, prediction_date, team_results, team_log)
    X_ab = row_ab[models["feature_cols"]]

    prob_ab = models["win_model"].predict_proba(X_ab)[0, 1]
    score1_ab = float(models["score1_model"].predict(X_ab)[0])
    score2_ab = float(models["score2_model"].predict(X_ab)[0])

    # reverse order
    row_ba = build_matchup_row(team_2, team_1, prediction_date, team_results, team_log)
    X_ba = row_ba[models["feature_cols"]]

    prob_ba_team2 = models["win_model"].predict_proba(X_ba)[0, 1]
    score1_ba = float(models["score1_model"].predict(X_ba)[0])  # original team_2
    score2_ba = float(models["score2_model"].predict(X_ba)[0])  # original team_1

    # convert reversed probability back to original team_1 perspective
    prob_ba_team1 = 1 - prob_ba_team2

    # neutral-site averages
    team_1_win_prob = (prob_ab + prob_ba_team1) / 2.0
    team_2_win_prob = 1 - team_1_win_prob

    pred_team_1_score = (score1_ab + score2_ba) / 2.0
    pred_team_2_score = (score2_ab + score1_ba) / 2.0

    pred_team_1_score = int(round(pred_team_1_score))
    pred_team_2_score = int(round(pred_team_2_score))

    if pred_team_1_score == pred_team_2_score:
        if team_1_win_prob >= 0.5:
            pred_team_1_score += 1
        else:
            pred_team_2_score += 1

    predicted_winner = team_1 if team_1_win_prob >= 0.5 else team_2

    return {
        "team_1": team_1,
        "team_2": team_2,
        "team_1_win_probability": team_1_win_prob,
        "team_2_win_probability": team_2_win_prob,
        "pred_team_1_score": pred_team_1_score,
        "pred_team_2_score": pred_team_2_score,
        "predicted_winner": predicted_winner,
        "predicted_margin": pred_team_1_score - pred_team_2_score,
        "predicted_total": pred_team_1_score + pred_team_2_score,
    }


def print_prediction(result: dict):
    print("\nNeutral-site matchup prediction")
    print("-------------------------------")
    print(f"{result['team_1']} vs {result['team_2']}")
    print(f"{result['team_1']}: {result['team_1_win_probability']:.3%} win probability")
    print(f"{result['team_2']}: {result['team_2_win_probability']:.3%} win probability")
    print(
        f"Projected score: {result['team_1']} {result['pred_team_1_score']} - "
        f"{result['team_2']} {result['pred_team_2_score']}"
    )
    print(f"Predicted winner: {result['predicted_winner']}")
    print(f"Predicted margin: {result['predicted_margin']}")
    print(f"Predicted total: {result['predicted_total']}")


def main():
    file_path = Path(EXCEL_FILE)
    if not file_path.exists():
        print(f"File not found: {file_path.resolve()}")
        sys.exit(1)

    schedule_df, name_match_df, team_results_df, sheet_names = load_workbook(EXCEL_FILE)

    print("Workbook loaded")
    print("Sheets:", sheet_names)

    name_map = build_name_map(name_match_df)
    games = prepare_games(schedule_df, name_map)
    team_results = prepare_team_results(team_results_df, name_map)
    team_log = build_team_game_log(games)
    model_df = build_training_dataset(games, team_results, team_log)

    model_df = model_df[
        model_df["team_1_adjoe"].notna() &
        model_df["team_2_adjoe"].notna()
    ].copy()

    print(f"\nSchedule rows since 2025-12-01: {len(games)}")
    print(f"Completed games: {int(games['is_completed'].sum())}")
    print(f"Upcoming games: {int((~games['is_completed']).sum())}")
    print(f"Team results rows: {len(team_results)}")
    print(f"Rows remaining for training: {len(model_df)}")

    models = train_models(model_df)

    prediction_date = games["game_date"].max() + pd.Timedelta(days=1)
    if pd.isna(prediction_date):
        prediction_date = pd.Timestamp.now(tz="UTC")

    print("\nReady for arbitrary neutral-site matchups.")
    print("Enter two teams. Type 'quit' to exit.")

    while True:
        team_1_input = input("\nTeam 1: ").strip()
        if not team_1_input or team_1_input.lower() in {"quit", "exit"}:
            break

        team_2_input = input("Team 2: ").strip()
        if not team_2_input or team_2_input.lower() in {"quit", "exit"}:
            break

        try:
            team_1 = canonicalize_input_team(team_1_input, name_map, team_results)
            team_2 = canonicalize_input_team(team_2_input, name_map, team_results)

            if team_1 == team_2:
                print("Please enter two different teams.")
                continue

            result = predict_matchup(
                team_1=team_1,
                team_2=team_2,
                models=models,
                team_results=team_results,
                team_log=team_log,
                prediction_date=prediction_date,
            )
            print_prediction(result)

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()