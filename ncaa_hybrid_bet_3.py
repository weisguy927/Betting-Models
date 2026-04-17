import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


EXCEL_FILE = "ncaa_schedule_compiled_26.xlsx"

SCHEDULE_SHEET = "Schedule"
NAME_MATCH_SHEET = "Name Match"
TEAM_RESULTS_SHEET = "2026_team_results"
HIST_ODDS_SHEET = "Historical Odds"

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

    required = [SCHEDULE_SHEET, NAME_MATCH_SHEET, TEAM_RESULTS_SHEET, HIST_ODDS_SHEET]
    missing = [s for s in required if s not in sheet_names]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}. Available: {sheet_names}")

    schedule_df = pd.read_excel(path, sheet_name=SCHEDULE_SHEET)
    name_match_df = pd.read_excel(path, sheet_name=NAME_MATCH_SHEET)
    team_results_df = pd.read_excel(path, sheet_name=TEAM_RESULTS_SHEET)
    historical_odds_df = pd.read_excel(path, sheet_name=HIST_ODDS_SHEET)

    return schedule_df, name_match_df, team_results_df, historical_odds_df, sheet_names


def build_name_maps(name_match_df: pd.DataFrame) -> tuple[dict, dict]:
    cols = list(name_match_df.columns)

    source_col = find_column(cols, [
        "source_name", "source", "raw_name", "original_name",
        "schedule_name", "alias", "team", "schedule team",
        "schedule_team", "name1"
    ])

    target_col_model = find_column(cols, [
        "matched_name", "standard_name", "canonical_name",
        "team_name", "standardized_name", "matched team",
        "matched_team", "name2"
    ])

    target_col_odds = find_column(cols, [
        "matched_name_2", "matched name 2", "odds_name",
        "historical_odds_name", "sportsbook_name"
    ])

    if source_col is None:
        raise ValueError(f"Could not identify source column in Name Match. Found: {cols}")

    if target_col_model is None:
        raise ValueError(f"Could not identify matched_name column in Name Match. Found: {cols}")

    if target_col_odds is None:
        raise ValueError(f"Could not identify matched_name_2 column in Name Match. Found: {cols}")

    model_map = {}
    odds_map = {}

    for _, row in name_match_df.iterrows():
        src = clean_name(row[source_col])
        tgt_model = clean_name(row[target_col_model])
        tgt_odds = clean_name(row[target_col_odds])

        if src and tgt_model:
            model_map[src] = tgt_model

        if tgt_odds and tgt_model:
            odds_map[tgt_odds] = tgt_model

    if not model_map:
        raise ValueError("No usable model mappings created.")
    if not odds_map:
        raise ValueError("No usable odds mappings created.")

    print(f"Loaded {len(model_map)} model mappings")
    print(f"Loaded {len(odds_map)} odds mappings")

    return model_map, odds_map


def standardize_team(name: str, mapping: dict) -> str:
    cleaned = clean_name(name)
    return mapping.get(cleaned, cleaned)


def prepare_games(schedule_df: pd.DataFrame, model_map: dict) -> pd.DataFrame:
    cols = list(schedule_df.columns)

    date_col = find_column(cols, ["date", "game_date", "start_date", "startdate"])
    team1_col = find_column(cols, ["home_team", "home team", "home", "team_home"])
    team2_col = find_column(cols, ["away_team", "away team", "away", "team_away"])
    score1_col = find_column(cols, ["home_score", "home score", "score_home"])
    score2_col = find_column(cols, ["away_score", "away score", "score_away"])
    status_col = find_column(cols, ["status", "game_state", "gamestate", "state"])

    missing = [
        name for name, col in {
            "date": date_col,
            "team_1": team1_col,
            "team_2": team2_col,
        }.items() if col is None
    ]
    if missing:
        raise ValueError(
            f"Missing required columns in Schedule sheet: {missing}. "
            f"Available columns: {cols}"
        )

    games = schedule_df.copy()
    games["game_date"] = pd.to_datetime(games[date_col], errors="coerce", utc=True)
    games["team_1"] = games[team1_col].apply(lambda x: standardize_team(x, model_map))
    games["team_2"] = games[team2_col].apply(lambda x: standardize_team(x, model_map))

    games["team_1_score"] = pd.to_numeric(games[score1_col], errors="coerce") if score1_col else np.nan
    games["team_2_score"] = pd.to_numeric(games[score2_col], errors="coerce") if score2_col else np.nan
    games["status_std"] = games[status_col].astype(str).str.lower().str.strip() if status_col else ""

    games["is_completed"] = games["team_1_score"].notna() & games["team_2_score"].notna()
    games["team_1_win"] = np.where(
        games["is_completed"],
        (games["team_1_score"] > games["team_2_score"]).astype(int),
        np.nan
    )

    games = games[games["game_date"] >= MODEL_START_DATE].copy()
    return games.sort_values("game_date").reset_index(drop=True)


def prepare_team_results(team_results_df: pd.DataFrame, model_map: dict) -> pd.DataFrame:
    df = team_results_df.copy()

    required_cols = ["team", "adjoe", "adjde", "barthag", "adjt", "sos", "WAB", "Conf Win%"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {TEAM_RESULTS_SHEET}: {missing}")

    df["team_std"] = df["team"].apply(lambda x: standardize_team(x, model_map))

    numeric_cols = ["adjoe", "adjde", "barthag", "adjt", "sos", "WAB", "Conf Win%"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["team_std"] + numeric_cols
    return df[keep_cols].drop_duplicates(subset=["team_std"])


def american_to_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig_two_way(prob_a, prob_b):
    if pd.isna(prob_a) or pd.isna(prob_b):
        return np.nan, np.nan
    total = prob_a + prob_b
    if total <= 0:
        return np.nan, np.nan
    return prob_a / total, prob_b / total


def prepare_historical_odds(historical_odds_df: pd.DataFrame, odds_map: dict) -> pd.DataFrame:
    df = historical_odds_df.copy()
    cols = list(df.columns)

    game_time_col = find_column(cols, ["game_time", "commence_time", "start_time"])
    home_team_col = find_column(cols, ["home_team", "home team"])
    away_team_col = find_column(cols, ["away_team", "away team"])
    book_col = find_column(cols, ["bookmaker", "sportsbook", "book"])
    ml_home_col = find_column(cols, ["moneyline_home", "home_moneyline", "ml_home"])
    ml_away_col = find_column(cols, ["moneyline_away", "away_moneyline", "ml_away"])
    spread_home_col = find_column(cols, ["spread_home", "home_spread"])
    spread_away_col = find_column(cols, ["spread_away", "away_spread"])
    total_over_col = find_column(cols, ["total_over", "over"])
    total_under_col = find_column(cols, ["total_under", "under"])

    required = {
        "game_time": game_time_col,
        "home_team": home_team_col,
        "away_team": away_team_col,
        "bookmaker": book_col,
        "moneyline_home": ml_home_col,
        "moneyline_away": ml_away_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required columns in Historical Odds sheet: {missing}. "
            f"Available columns: {cols}"
        )

    df["game_time"] = pd.to_datetime(df[game_time_col], errors="coerce", utc=True)

    df["home_team_std"] = df[home_team_col].apply(
        lambda x: odds_map.get(clean_name(x), clean_name(x))
    )
    df["away_team_std"] = df[away_team_col].apply(
        lambda x: odds_map.get(clean_name(x), clean_name(x))
    )

    df["bookmaker"] = df[book_col].astype(str)

    df["moneyline_home"] = pd.to_numeric(df[ml_home_col], errors="coerce")
    df["moneyline_away"] = pd.to_numeric(df[ml_away_col], errors="coerce")
    df["spread_home"] = pd.to_numeric(df[spread_home_col], errors="coerce") if spread_home_col else np.nan
    df["spread_away"] = pd.to_numeric(df[spread_away_col], errors="coerce") if spread_away_col else np.nan

    df["total_over"] = pd.to_numeric(df[total_over_col], errors="coerce") if total_over_col else np.nan
    df["total_under"] = pd.to_numeric(df[total_under_col], errors="coerce") if total_under_col else np.nan

    df["total_points"] = df[["total_over", "total_under"]].mean(axis=1)
    df = df[df["game_time"] >= MODEL_START_DATE].copy()

    return df.reset_index(drop=True)


def get_latest_pregame_market(odds_df: pd.DataFrame) -> pd.DataFrame:
    latest = odds_df.copy()

    latest["implied_prob_home_raw"] = latest["moneyline_home"].apply(american_to_implied_prob)
    latest["implied_prob_away_raw"] = latest["moneyline_away"].apply(american_to_implied_prob)

    fair_probs = latest.apply(
        lambda row: remove_vig_two_way(
            row["implied_prob_home_raw"],
            row["implied_prob_away_raw"]
        ),
        axis=1
    )

    latest["fair_prob_home"] = [x[0] for x in fair_probs]
    latest["fair_prob_away"] = [x[1] for x in fair_probs]
    latest["market_hold"] = latest["implied_prob_home_raw"] + latest["implied_prob_away_raw"] - 1

    market = (
        latest.groupby(["home_team_std", "away_team_std", "game_time"], as_index=False)
        .agg(
            market_prob_team_1=("fair_prob_home", "mean"),
            market_prob_team_2=("fair_prob_away", "mean"),
            market_ml_team_1=("moneyline_home", "mean"),
            market_ml_team_2=("moneyline_away", "mean"),
            market_spread_team_1=("spread_home", "mean"),
            market_spread_team_2=("spread_away", "mean"),
            market_total=("total_points", "mean"),
            market_hold=("market_hold", "mean"),
            bookmakers_count=("bookmaker", "nunique"),
        )
    )

    market["market_prob_diff"] = market["market_prob_team_1"] - market["market_prob_team_2"]
    market["game_date_only"] = market["game_time"].dt.normalize()

    return market


def build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    completed = games[games["is_completed"]].copy()

    team1_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_1"],
        "opponent": completed["team_2"],
        "points_for": completed["team_1_score"],
        "points_against": completed["team_2_score"],
        "win": (completed["team_1_score"] > completed["team_2_score"]).astype(int),
    })

    team2_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_2"],
        "opponent": completed["team_1"],
        "points_for": completed["team_2_score"],
        "points_against": completed["team_1_score"],
        "win": (completed["team_2_score"] > completed["team_1_score"]).astype(int),
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
        return default

    last = rows.sort_values("game_date").iloc[-1]
    result = {"games_played_prior": int(last["games_played_prior"]) if pd.notna(last["games_played_prior"]) else 0}

    for window in FORM_WINDOWS:
        result[f"recent_win_pct_{window}"] = float(last[f"rolling_win_pct_{window}"]) if pd.notna(last[f"rolling_win_pct_{window}"]) else 0.5
        result[f"recent_points_for_{window}"] = float(last[f"rolling_points_for_{window}"]) if pd.notna(last[f"rolling_points_for_{window}"]) else 70.0
        result[f"recent_points_against_{window}"] = float(last[f"rolling_points_against_{window}"]) if pd.notna(last[f"rolling_points_against_{window}"]) else 70.0
        result[f"recent_scoring_margin_{window}"] = float(last[f"rolling_scoring_margin_{window}"]) if pd.notna(last[f"rolling_scoring_margin_{window}"]) else 0.0

    return result


def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["adjoe_diff"] = df["team_1_adjoe"] - df["team_2_adjoe"]
    df["adjde_diff"] = df["team_1_adjde"] - df["team_2_adjde"]
    df["barthag_diff"] = df["team_1_barthag"] - df["team_2_barthag"]
    df["adjt_diff"] = df["team_1_adjt"] - df["team_2_adjt"]
    df["sos_diff"] = df["team_1_sos"] - df["team_2_sos"]
    df["WAB_diff"] = df["team_1_WAB"] - df["team_2_WAB"]
    df["conf_win_pct_diff"] = df["team_1_conf_win_pct"] - df["team_2_conf_win_pct"]
    df["games_played_prior_diff"] = df["team_1_games_played_prior"] - df["team_2_games_played_prior"]

    for window in FORM_WINDOWS:
        df[f"recent_win_pct_diff_{window}"] = df[f"team_1_recent_win_pct_{window}"] - df[f"team_2_recent_win_pct_{window}"]
        df[f"recent_points_for_diff_{window}"] = df[f"team_1_recent_points_for_{window}"] - df[f"team_2_recent_points_for_{window}"]
        df[f"recent_points_against_diff_{window}"] = df[f"team_1_recent_points_against_{window}"] - df[f"team_2_recent_points_against_{window}"]
        df[f"recent_scoring_margin_diff_{window}"] = df[f"team_1_recent_scoring_margin_{window}"] - df[f"team_2_recent_scoring_margin_{window}"]

    return df


def build_base_model_dataset(games: pd.DataFrame, team_results: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
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
    return add_difference_features(model_df)


def add_market_features(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.copy()
    df["game_date_only"] = df["game_date"].dt.normalize()

    merged = df.merge(
        market_df,
        left_on=["team_1", "team_2", "game_date_only"],
        right_on=["home_team_std", "away_team_std", "game_date_only"],
        how="left"
    )

    return merged


def print_unmatched_summary(model_df: pd.DataFrame):
    missing_team1 = model_df["team_1_adjoe"].isna().sum()
    missing_team2 = model_df["team_2_adjoe"].isna().sum()

    print(f"\nRows missing team_1 team-results stats: {missing_team1}")
    print(f"Rows missing team_2 team-results stats: {missing_team2}")

    unmatched_team2 = model_df.loc[model_df["team_2_adjoe"].isna(), "team_2"].value_counts().head(50)
    unmatched_team1 = model_df.loc[model_df["team_1_adjoe"].isna(), "team_1"].value_counts().head(50)

    print("\nMost common unmatched team_2 names:")
    print("None" if unmatched_team2.empty else unmatched_team2.to_string())

    print("\nMost common unmatched team_1 names:")
    print("None" if unmatched_team1.empty else unmatched_team1.to_string())


def filter_to_games_with_required_data(model_df: pd.DataFrame) -> pd.DataFrame:
    pre_rows = len(model_df)

    filtered = model_df[
        model_df["team_1_adjoe"].notna() &
        model_df["team_2_adjoe"].notna() &
        model_df["market_prob_team_1"].notna() &
        model_df["market_prob_team_2"].notna()
    ].copy()

    removed = pre_rows - len(filtered)

    print(f"\nFiltered out {removed} rows missing team ratings or market odds.")
    print(f"Remaining rows for betting model: {len(filtered)}")
    return filtered


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
        ("market_prob_team_1", "market_prob_team_2"),
        ("market_ml_team_1", "market_ml_team_2"),
        ("market_spread_team_1", "market_spread_team_2"),
    ]

    for window in FORM_WINDOWS:
        swap_pairs.extend([
            (f"team_1_recent_win_pct_{window}", f"team_2_recent_win_pct_{window}"),
            (f"team_1_recent_points_for_{window}", f"team_2_recent_points_for_{window}"),
            (f"team_1_recent_points_against_{window}", f"team_2_recent_points_against_{window}"),
            (f"team_1_recent_scoring_margin_{window}", f"team_2_recent_scoring_margin_{window}"),
        ])

    for left_col, right_col in swap_pairs:
        if left_col in swapped.columns and right_col in swapped.columns:
            swapped[left_col], swapped[right_col] = original[right_col], original[left_col]

    swapped["team_1_win"] = 1 - original["team_1_win"].astype(int)
    swapped["market_prob_diff"] = swapped["market_prob_team_1"] - swapped["market_prob_team_2"]

    swapped = add_difference_features(swapped)

    symmetric = pd.concat([original, swapped], ignore_index=True)
    return symmetric.sort_values(["game_date", "team_1", "team_2"]).reset_index(drop=True)


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

        "market_prob_team_1", "market_prob_team_2", "market_prob_diff",
        "market_ml_team_1", "market_ml_team_2",
        "market_spread_team_1", "market_spread_team_2",
        "market_total", "market_hold", "bookmakers_count",
    ]

    for side in ["team_1", "team_2"]:
        for window in FORM_WINDOWS:
            feature_cols.extend([
                f"{side}_recent_win_pct_{window}",
                f"{side}_recent_points_for_{window}",
                f"{side}_recent_points_against_{window}",
                f"{side}_recent_scoring_margin_{window}",
            ])

    for window in FORM_WINDOWS:
        feature_cols.extend([
            f"recent_win_pct_diff_{window}",
            f"recent_points_for_diff_{window}",
            f"recent_points_against_diff_{window}",
            f"recent_scoring_margin_diff_{window}",
        ])

    return feature_cols


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


def find_best_threshold(y_true: pd.Series, probs: np.ndarray):
    thresholds = np.arange(0.35, 0.66, 0.01)
    best_threshold = 0.50
    best_accuracy = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def moneyline_profit_units(odds, won, stake=1.0):
    if pd.isna(odds):
        return 0.0
    odds = float(odds)
    if won:
        if odds > 0:
            return stake * (odds / 100.0)
        return stake * (100.0 / abs(odds))
    return -stake


def build_bet_df(test_df: pd.DataFrame, probs: np.ndarray, edge_threshold: float = 0.03, label: str = "full_model"):
    df = test_df.copy()
    df["model_prob_team_1"] = probs
    df["model_prob_team_2"] = 1 - probs

    df["edge_team_1"] = df["model_prob_team_1"] - df["market_prob_team_1"]
    df["edge_team_2"] = df["model_prob_team_2"] - df["market_prob_team_2"]

    bet_rows = []

    for _, row in df.iterrows():
        if pd.isna(row["market_ml_team_1"]) or pd.isna(row["market_ml_team_2"]):
            continue

        if row["edge_team_1"] >= edge_threshold and row["edge_team_1"] >= row["edge_team_2"]:
            won = int(row["team_1_win"] == 1)
            profit = moneyline_profit_units(row["market_ml_team_1"], won)
            bet_rows.append({
                "model_label": label,
                "game_date": row["game_date"],
                "bet_side": "team_1",
                "team": row["team_1"],
                "opponent": row["team_2"],
                "model_prob": row["model_prob_team_1"],
                "market_prob": row["market_prob_team_1"],
                "edge": row["edge_team_1"],
                "odds": row["market_ml_team_1"],
                "won": won,
                "profit_units": profit,
            })

        elif row["edge_team_2"] >= edge_threshold:
            won = int(row["team_1_win"] == 0)
            profit = moneyline_profit_units(row["market_ml_team_2"], won)
            bet_rows.append({
                "model_label": label,
                "game_date": row["game_date"],
                "bet_side": "team_2",
                "team": row["team_2"],
                "opponent": row["team_1"],
                "model_prob": row["model_prob_team_2"],
                "market_prob": row["market_prob_team_2"],
                "edge": row["edge_team_2"],
                "odds": row["market_ml_team_2"],
                "won": won,
                "profit_units": profit,
            })

    return pd.DataFrame(bet_rows)


def summarize_bets(bets_df: pd.DataFrame, edge_threshold: float):
    n_bets = len(bets_df)
    profit = bets_df["profit_units"].sum() if n_bets else 0.0
    roi = profit / n_bets if n_bets else 0.0
    win_rate = bets_df["won"].mean() if n_bets else 0.0

    return {
        "edge_threshold": edge_threshold,
        "bets": n_bets,
        "profit_units": profit,
        "roi": roi,
        "win_rate": win_rate,
        "bets_df": bets_df,
    }


def backtest_moneyline_bets(test_df: pd.DataFrame, probs: np.ndarray, edge_threshold: float = 0.03, label: str = "full_model"):
    bets_df = build_bet_df(test_df, probs, edge_threshold=edge_threshold, label=label)
    return summarize_bets(bets_df, edge_threshold=edge_threshold)


def sweep_edge_thresholds(test_df: pd.DataFrame, probs: np.ndarray, label: str = "full_model"):
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06]
    results = []

    print(f"\nBetting backtest threshold sweep ({label})")
    print("----------------------------------------")

    for threshold in thresholds:
        result = backtest_moneyline_bets(test_df, probs, edge_threshold=threshold, label=label)
        results.append({
            "model_label": label,
            "edge_threshold": result["edge_threshold"],
            "bets": result["bets"],
            "profit_units": result["profit_units"],
            "roi": result["roi"],
            "win_rate": result["win_rate"],
        })

    results_df = pd.DataFrame(results).sort_values("edge_threshold").reset_index(drop=True)
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["roi"].idxmax()]
    best_threshold = float(best_row["edge_threshold"])

    print(f"\nBest betting threshold by ROI ({label}): {best_threshold:.2f}")

    best_result = backtest_moneyline_bets(test_df, probs, edge_threshold=best_threshold, label=label)

    print(f"\nBest-threshold betting backtest ({label})")
    print("----------------------------------------")
    print(f"Edge threshold: {best_result['edge_threshold']:.3f}")
    print(f"Bets placed: {best_result['bets']}")
    print(f"Profit (units): {best_result['profit_units']:.2f}")
    print(f"ROI: {best_result['roi']:.3%}")
    print(f"Win rate: {best_result['win_rate']:.3%}")

    return results_df, best_result


def monthly_backtest_summary(bets_df: pd.DataFrame, label: str):
    if bets_df.empty:
        return pd.DataFrame(columns=["model_label", "month", "bets", "profit_units", "roi", "win_rate"])

    df = bets_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    df["month"] = df["game_date"].dt.to_period("M").astype(str)

    summary = (
        df.groupby("month", as_index=False)
        .agg(
            bets=("profit_units", "size"),
            profit_units=("profit_units", "sum"),
            win_rate=("won", "mean"),
        )
    )
    summary["roi"] = summary["profit_units"] / summary["bets"]
    summary["model_label"] = label

    return summary[["model_label", "month", "bets", "profit_units", "roi", "win_rate"]]


def calibration_buckets(test_df: pd.DataFrame, probs: np.ndarray, label: str):
    df = test_df.copy()
    df["pred_prob"] = probs
    df["actual"] = df["team_1_win"].astype(int)

    bins = [0.0, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    labels = ["<=55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"]

    df["prob_bucket"] = pd.cut(df["pred_prob"], bins=bins, labels=labels, include_lowest=True)

    calib = (
        df.groupby("prob_bucket", observed=False, as_index=False)
        .agg(
            n=("actual", "size"),
            avg_pred_prob=("pred_prob", "mean"),
            actual_win_rate=("actual", "mean"),
        )
    )
    calib["calibration_gap"] = calib["actual_win_rate"] - calib["avg_pred_prob"]
    calib["model_label"] = label

    return calib[["model_label", "prob_bucket", "n", "avg_pred_prob", "actual_win_rate", "calibration_gap"]]


def train_market_only_baseline(train_part: pd.DataFrame, test_part: pd.DataFrame):
    baseline_features = [
        "market_prob_team_1",
        "market_prob_team_2",
        "market_prob_diff",
        "market_ml_team_1",
        "market_ml_team_2",
        "market_spread_team_1",
        "market_spread_team_2",
        "market_total",
        "market_hold",
        "bookmakers_count",
    ]

    train_df = train_part.dropna(subset=baseline_features + ["team_1_win"]).copy()
    test_df = test_part.dropna(subset=baseline_features + ["team_1_win"]).copy()

    X_train = train_df[baseline_features]
    y_train = train_df["team_1_win"].astype(int)

    X_test = test_df[baseline_features]
    y_test = test_df["team_1_win"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print("\nMarket-only baseline")
    print("--------------------")
    print(f"Rows used: train={len(train_df)}, test={len(test_df)}")
    print(f"ROC AUC: {auc:.3f}")

    return model, baseline_features, probs, test_df


def train_predictive_model(model_df: pd.DataFrame):
    feature_cols = build_feature_columns()

    completed_df = model_df[model_df["is_completed"]].copy()
    completed_df = completed_df.dropna(subset=feature_cols + ["team_1_win", "game_date"])

    if len(completed_df) < 20:
        raise ValueError(f"Not enough completed games to train model. Found {len(completed_df)} rows.")

    symmetric_df = make_symmetric_training_dataset(completed_df)
    symmetric_df = symmetric_df.sort_values("game_date").reset_index(drop=True)

    train_part, test_part, split_date = time_based_split(symmetric_df)

    X_train = train_part[feature_cols]
    y_train = train_part["team_1_win"].astype(int)

    X_test = test_part[feature_cols]
    y_test = test_part["team_1_win"].astype(int)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    best_threshold, best_accuracy = find_best_threshold(y_test, probs)
    preds = (probs >= best_threshold).astype(int)

    print("\nModel evaluation")
    print("----------------")
    print(f"Original completed rows used: {len(completed_df)}")
    print(f"Symmetric completed rows used: {len(symmetric_df)}")
    print(f"Train rows: {len(train_part)}")
    print(f"Test rows: {len(test_part)}")
    print(f"Time split date: {pd.Timestamp(split_date).date()}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"Best-threshold accuracy: {best_accuracy:.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")

    print("\nClassification report")
    print(classification_report(y_test, preds))

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances")
    print(importances.head(25).to_string())

    baseline_model, baseline_features, baseline_probs, baseline_test_df = train_market_only_baseline(train_part, test_part)

    threshold_results_df, best_bet_result = sweep_edge_thresholds(test_part, probs, label="full_model")
    baseline_threshold_results_df, baseline_best_bet_result = sweep_edge_thresholds(
        baseline_test_df, baseline_probs, label="market_only"
    )

    monthly_full_df = monthly_backtest_summary(best_bet_result["bets_df"], label="full_model")
    monthly_baseline_df = monthly_backtest_summary(baseline_best_bet_result["bets_df"], label="market_only")

    calibration_full_df = calibration_buckets(test_part, probs, label="full_model")
    calibration_baseline_df = calibration_buckets(baseline_test_df, baseline_probs, label="market_only")

    print("\nMonthly backtest summary (full model)")
    print("------------------------------------")
    print("No bets" if monthly_full_df.empty else monthly_full_df.to_string(index=False))

    print("\nMonthly backtest summary (market only)")
    print("-------------------------------------")
    print("No bets" if monthly_baseline_df.empty else monthly_baseline_df.to_string(index=False))

    print("\nCalibration buckets (full model)")
    print("-------------------------------")
    print(calibration_full_df.to_string(index=False))

    print("\nCalibration buckets (market only)")
    print("--------------------------------")
    print(calibration_baseline_df.to_string(index=False))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "split_date": split_date,
        "best_threshold": best_threshold,
        "test_part": test_part,
        "test_probs": probs,
        "threshold_results_df": threshold_results_df,
        "best_bet_result": best_bet_result,
        "baseline_model": baseline_model,
        "baseline_features": baseline_features,
        "baseline_probs": baseline_probs,
        "baseline_test_df": baseline_test_df,
        "baseline_threshold_results_df": baseline_threshold_results_df,
        "baseline_best_bet_result": baseline_best_bet_result,
        "monthly_full_df": monthly_full_df,
        "monthly_baseline_df": monthly_baseline_df,
        "calibration_full_df": calibration_full_df,
        "calibration_baseline_df": calibration_baseline_df,
    }


def predict_upcoming_games(model_df: pd.DataFrame, model, feature_cols, threshold: float, edge_threshold: float) -> pd.DataFrame:
    pred_df = model_df[~model_df["is_completed"]].copy()
    pred_df = pred_df.dropna(subset=feature_cols)

    if pred_df.empty:
        print("\nNo upcoming games available for prediction.")
        return pred_df

    pred_df["team_1_win_probability"] = model.predict_proba(pred_df[feature_cols])[:, 1]
    pred_df["team_2_win_probability"] = 1 - pred_df["team_1_win_probability"]

    pred_df["edge_team_1"] = pred_df["team_1_win_probability"] - pred_df["market_prob_team_1"]
    pred_df["edge_team_2"] = pred_df["team_2_win_probability"] - pred_df["market_prob_team_2"]

    pred_df["predicted_winner"] = np.where(
        pred_df["team_1_win_probability"] >= 0.5,
        pred_df["team_1"],
        pred_df["team_2"],
    )

    pred_df["threshold_pick"] = np.where(
        pred_df["team_1_win_probability"] >= threshold,
        pred_df["team_1"],
        pred_df["team_2"],
    )

    pred_df["bet_recommendation"] = np.select(
        [
            (pred_df["edge_team_1"] >= edge_threshold) & (pred_df["edge_team_1"] >= pred_df["edge_team_2"]),
            pred_df["edge_team_2"] >= edge_threshold,
        ],
        [
            "Bet team_1 ML",
            "Bet team_2 ML",
        ],
        default="No bet"
    )

    pred_df["confidence_gap"] = (pred_df["team_1_win_probability"] - 0.5).abs()
    pred_df["confidence_tier"] = pd.cut(
        pred_df["confidence_gap"],
        bins=[0.0, 0.05, 0.10, 0.20, 1.0],
        labels=["coin flip", "slight edge", "moderate edge", "strong edge"],
        include_lowest=True
    )

    return pred_df.sort_values("game_date").reset_index(drop=True)


def main():
    file_path = Path(EXCEL_FILE)
    if not file_path.exists():
        print(f"File not found: {file_path.resolve()}")
        sys.exit(1)

    schedule_df, name_match_df, team_results_df, historical_odds_df, sheet_names = load_workbook(EXCEL_FILE)

    print("Workbook loaded")
    print("Sheets:", sheet_names)

    model_map, odds_map = build_name_maps(name_match_df)
    games = prepare_games(schedule_df, model_map)
    team_results = prepare_team_results(team_results_df, model_map)
    historical_odds = prepare_historical_odds(historical_odds_df, odds_map)
    market_df = get_latest_pregame_market(historical_odds)

    print(f"\nSchedule rows since 2025-12-01: {len(games)}")
    print(f"Completed games: {int(games['is_completed'].sum())}")
    print(f"Upcoming games: {int((~games['is_completed']).sum())}")
    print(f"Team results rows: {len(team_results)}")
    print(f"Historical odds rows: {len(historical_odds)}")
    print(f"Latest market rows: {len(market_df)}")

    team_log = build_team_game_log(games)
    model_df = build_base_model_dataset(games, team_results, team_log)
    print_unmatched_summary(model_df)

    model_df = add_market_features(model_df, market_df)
    model_df = filter_to_games_with_required_data(model_df)

    results = train_predictive_model(model_df)

    model = results["model"]
    feature_cols = results["feature_cols"]
    best_threshold = results["best_threshold"]

    threshold_results_df = results["threshold_results_df"]
    baseline_threshold_results_df = results["baseline_threshold_results_df"]

    best_bet_result = results["best_bet_result"]
    baseline_best_bet_result = results["baseline_best_bet_result"]

    monthly_full_df = results["monthly_full_df"]
    monthly_baseline_df = results["monthly_baseline_df"]

    calibration_full_df = results["calibration_full_df"]
    calibration_baseline_df = results["calibration_baseline_df"]

    bets_df = best_bet_result["bets_df"]
    baseline_bets_df = baseline_best_bet_result["bets_df"]

    predictions = predict_upcoming_games(
        model_df,
        model,
        feature_cols,
        best_threshold,
        edge_threshold=float(best_bet_result["edge_threshold"]),
    )

    if not threshold_results_df.empty:
        threshold_results_df.to_csv("betting_threshold_sweep_results_full_model.csv", index=False)
        print("\nSaved betting_threshold_sweep_results_full_model.csv")

    if not baseline_threshold_results_df.empty:
        baseline_threshold_results_df.to_csv("betting_threshold_sweep_results_market_only.csv", index=False)
        print("Saved betting_threshold_sweep_results_market_only.csv")

    if not bets_df.empty:
        bets_df.to_csv("betting_model_backtest_bets_full_model.csv", index=False)
        print("Saved betting_model_backtest_bets_full_model.csv")

    if not baseline_bets_df.empty:
        baseline_bets_df.to_csv("betting_model_backtest_bets_market_only.csv", index=False)
        print("Saved betting_model_backtest_bets_market_only.csv")

    if not monthly_full_df.empty:
        monthly_full_df.to_csv("monthly_backtest_full_model.csv", index=False)
        print("Saved monthly_backtest_full_model.csv")

    if not monthly_baseline_df.empty:
        monthly_baseline_df.to_csv("monthly_backtest_market_only.csv", index=False)
        print("Saved monthly_backtest_market_only.csv")

    calibration_full_df.to_csv("calibration_buckets_full_model.csv", index=False)
    print("Saved calibration_buckets_full_model.csv")

    calibration_baseline_df.to_csv("calibration_buckets_market_only.csv", index=False)
    print("Saved calibration_buckets_market_only.csv")

    if not predictions.empty:
        output_cols = [
            "game_date",
            "team_1",
            "team_2",
            "team_1_win_probability",
            "team_2_win_probability",
            "market_prob_team_1",
            "market_prob_team_2",
            "edge_team_1",
            "edge_team_2",
            "predicted_winner",
            "threshold_pick",
            "bet_recommendation",
            "confidence_tier",
        ]
        predictions[output_cols].to_csv("betting_model_upcoming_predictions.csv", index=False)
        print("\nUpcoming predictions")
        print(predictions[output_cols].head(25).to_string(index=False))
        print("\nSaved betting_model_upcoming_predictions.csv")
    else:
        print("\nNo upcoming predictions generated.")


if __name__ == "__main__":
    main()