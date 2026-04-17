import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    if source_col is None or target_col_model is None or target_col_odds is None:
        raise ValueError(f"Could not identify required Name Match columns. Found: {cols}")

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

    missing = [
        name for name, col in {
            "date": date_col,
            "team_1": team1_col,
            "team_2": team2_col,
        }.items() if col is None
    ]
    if missing:
        raise ValueError(f"Missing required columns in Schedule sheet: {missing}. Available: {cols}")

    games = schedule_df.copy()
    games["game_date"] = pd.to_datetime(games[date_col], errors="coerce", utc=True)

    games["team_1"] = games[team1_col].apply(lambda x: standardize_team(x, model_map))
    games["team_2"] = games[team2_col].apply(lambda x: standardize_team(x, model_map))

    games["team_1_score"] = pd.to_numeric(games[score1_col], errors="coerce")
    games["team_2_score"] = pd.to_numeric(games[score2_col], errors="coerce")

    games["is_completed"] = games["team_1_score"].notna() & games["team_2_score"].notna()
    games["actual_total"] = games["team_1_score"] + games["team_2_score"]

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


def prepare_historical_odds(historical_odds_df: pd.DataFrame, odds_map: dict) -> pd.DataFrame:
    df = historical_odds_df.copy()
    cols = list(df.columns)

    game_time_col = find_column(cols, ["game_time", "commence_time", "start_time"])
    home_team_col = find_column(cols, ["home_team", "home team"])
    away_team_col = find_column(cols, ["away_team", "away team"])
    book_col = find_column(cols, ["bookmaker", "sportsbook", "book"])
    total_over_col = find_column(cols, ["total_over", "over"])
    total_under_col = find_column(cols, ["total_under", "under"])

    required = {
        "game_time": game_time_col,
        "home_team": home_team_col,
        "away_team": away_team_col,
        "bookmaker": book_col,
        "total_over": total_over_col,
        "total_under": total_under_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in Historical Odds sheet: {missing}. Available: {cols}")

    df["game_time"] = pd.to_datetime(df[game_time_col], errors="coerce", utc=True)
    df["home_team_std"] = df[home_team_col].apply(lambda x: odds_map.get(clean_name(x), clean_name(x)))
    df["away_team_std"] = df[away_team_col].apply(lambda x: odds_map.get(clean_name(x), clean_name(x)))
    df["bookmaker"] = df[book_col].astype(str)

    df["total_over"] = pd.to_numeric(df[total_over_col], errors="coerce")
    df["total_under"] = pd.to_numeric(df[total_under_col], errors="coerce")
    df["market_total"] = df[["total_over", "total_under"]].mean(axis=1)

    df = df[df["game_time"] >= MODEL_START_DATE].copy()
    return df.reset_index(drop=True)


def get_market_totals(odds_df: pd.DataFrame) -> pd.DataFrame:
    market = (
        odds_df.groupby(["home_team_std", "away_team_std", "game_time"], as_index=False)
        .agg(
            market_total=("market_total", "mean"),
            bookmakers_count=("bookmaker", "nunique"),
        )
    )
    market["game_date_only"] = market["game_time"].dt.normalize()
    return market


def build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    completed = games[games["is_completed"]].copy()

    team1_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_1"],
        "points_for": completed["team_1_score"],
        "points_against": completed["team_2_score"],
        "game_total": completed["actual_total"],
    })

    team2_rows = pd.DataFrame({
        "game_date": completed["game_date"],
        "team": completed["team_2"],
        "points_for": completed["team_2_score"],
        "points_against": completed["team_1_score"],
        "game_total": completed["actual_total"],
    })

    team_log = pd.concat([team1_rows, team2_rows], ignore_index=True)
    team_log["scoring_margin"] = team_log["points_for"] - team_log["points_against"]
    team_log = team_log.sort_values(["team", "game_date"]).reset_index(drop=True)

    for window in FORM_WINDOWS:
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
            default[f"recent_points_for_{window}"] = 70.0
            default[f"recent_points_against_{window}"] = 70.0
            default[f"recent_scoring_margin_{window}"] = 0.0
            default[f"recent_game_total_{window}"] = 140.0
        return default

    last = rows.sort_values("game_date").iloc[-1]
    result = {"games_played_prior": int(last["games_played_prior"]) if pd.notna(last["games_played_prior"]) else 0}

    for window in FORM_WINDOWS:
        result[f"recent_points_for_{window}"] = float(last[f"rolling_points_for_{window}"]) if pd.notna(last[f"rolling_points_for_{window}"]) else 70.0
        result[f"recent_points_against_{window}"] = float(last[f"rolling_points_against_{window}"]) if pd.notna(last[f"rolling_points_against_{window}"]) else 70.0
        result[f"recent_scoring_margin_{window}"] = float(last[f"rolling_scoring_margin_{window}"]) if pd.notna(last[f"rolling_scoring_margin_{window}"]) else 0.0
        result[f"recent_game_total_{window}"] = float(last[f"rolling_game_total_{window}"]) if pd.notna(last[f"rolling_game_total_{window}"]) else 140.0

    return result


def build_totals_regression_dataset(
    games: pd.DataFrame,
    team_results: pd.DataFrame,
    team_log: pd.DataFrame,
    totals_market_df: pd.DataFrame
) -> pd.DataFrame:
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

    model_df["game_date_only"] = model_df["game_date"].dt.normalize()
    model_df = model_df.merge(
        totals_market_df,
        left_on=["team_1", "team_2", "game_date_only"],
        right_on=["home_team_std", "away_team_std", "game_date_only"],
        how="left"
    )

    # stronger pace-based features
    model_df["combined_adjoe"] = model_df["team_1_adjoe"] + model_df["team_2_adjoe"]
    model_df["combined_adjde"] = model_df["team_1_adjde"] + model_df["team_2_adjde"]
    model_df["combined_adjt"] = model_df["team_1_adjt"] + model_df["team_2_adjt"]

    model_df["avg_adjt"] = (model_df["team_1_adjt"] + model_df["team_2_adjt"]) / 2.0
    model_df["tempo_gap"] = (model_df["team_1_adjt"] - model_df["team_2_adjt"]).abs()
    model_df["offense_vs_defense_sum"] = (
        model_df["team_1_adjoe"] + model_df["team_2_adjoe"] +
        model_df["team_1_adjde"] + model_df["team_2_adjde"]
    )
    model_df["offense_minus_defense_env"] = (
        (model_df["team_1_adjoe"] - model_df["team_2_adjde"]) +
        (model_df["team_2_adjoe"] - model_df["team_1_adjde"])
    )

    for window in FORM_WINDOWS:
        model_df[f"combined_recent_points_for_{window}"] = (
            model_df[f"team_1_recent_points_for_{window}"] + model_df[f"team_2_recent_points_for_{window}"]
        )
        model_df[f"combined_recent_points_against_{window}"] = (
            model_df[f"team_1_recent_points_against_{window}"] + model_df[f"team_2_recent_points_against_{window}"]
        )
        model_df[f"combined_recent_scoring_margin_{window}"] = (
            model_df[f"team_1_recent_scoring_margin_{window}"] + model_df[f"team_2_recent_scoring_margin_{window}"]
        )
        model_df[f"combined_recent_game_total_{window}"] = (
            model_df[f"team_1_recent_game_total_{window}"] + model_df[f"team_2_recent_game_total_{window}"]
        ) / 2.0

        model_df[f"pace_proxy_{window}"] = (
            model_df[f"combined_recent_game_total_{window}"] / np.maximum(model_df["avg_adjt"], 1e-6)
        )

    model_df["market_total_diff_from_recent_5"] = model_df["market_total"] - model_df["combined_recent_game_total_5"]
    model_df["market_total_diff_from_recent_10"] = model_df["market_total"] - model_df["combined_recent_game_total_10"]

    return model_df


def build_feature_columns():
    feature_cols = [
        "team_1_adjoe", "team_1_adjde", "team_1_barthag", "team_1_adjt",
        "team_2_adjoe", "team_2_adjde", "team_2_barthag", "team_2_adjt",
        "combined_adjoe", "combined_adjde", "combined_adjt",
        "avg_adjt", "tempo_gap",
        "offense_vs_defense_sum", "offense_minus_defense_env",
        "market_total", "bookmakers_count",
        "market_total_diff_from_recent_5",
        "market_total_diff_from_recent_10",
    ]

    for side in ["team_1", "team_2"]:
        for window in FORM_WINDOWS:
            feature_cols.extend([
                f"{side}_recent_points_for_{window}",
                f"{side}_recent_points_against_{window}",
                f"{side}_recent_scoring_margin_{window}",
                f"{side}_recent_game_total_{window}",
            ])

    for window in FORM_WINDOWS:
        feature_cols.extend([
            f"combined_recent_points_for_{window}",
            f"combined_recent_points_against_{window}",
            f"combined_recent_scoring_margin_{window}",
            f"combined_recent_game_total_{window}",
            f"pace_proxy_{window}",
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

    return train_part, test_part, split_date


def totals_profit_units(won, stake=1.0, american_odds=-110):
    if won:
        if american_odds > 0:
            return stake * (american_odds / 100.0)
        return stake * (100.0 / abs(american_odds))
    return -stake


def backtest_totals_bets(test_df: pd.DataFrame, predicted_totals: np.ndarray, line_edge_threshold: float = 3.0):
    df = test_df.copy()
    df["predicted_total_points"] = predicted_totals
    df["line_edge"] = df["predicted_total_points"] - df["market_total"]

    bet_rows = []
    total_profit = 0.0

    for _, row in df.iterrows():
        if pd.isna(row["actual_total"]) or pd.isna(row["market_total"]):
            continue

        if row["line_edge"] >= line_edge_threshold:
            won = int(row["actual_total"] > row["market_total"])
            profit = totals_profit_units(won, american_odds=-110)
            total_profit += profit
            bet_rows.append({
                "game_date": row["game_date"],
                "bet_side": "over",
                "team_1": row["team_1"],
                "team_2": row["team_2"],
                "market_total": row["market_total"],
                "predicted_total_points": row["predicted_total_points"],
                "line_edge": row["line_edge"],
                "actual_total": row["actual_total"],
                "won": won,
                "profit_units": profit,
            })

        elif row["line_edge"] <= -line_edge_threshold:
            won = int(row["actual_total"] < row["market_total"])
            profit = totals_profit_units(won, american_odds=-110)
            total_profit += profit
            bet_rows.append({
                "game_date": row["game_date"],
                "bet_side": "under",
                "team_1": row["team_1"],
                "team_2": row["team_2"],
                "market_total": row["market_total"],
                "predicted_total_points": row["predicted_total_points"],
                "line_edge": row["line_edge"],
                "actual_total": row["actual_total"],
                "won": won,
                "profit_units": profit,
            })

    bets_df = pd.DataFrame(bet_rows)
    n_bets = len(bets_df)
    roi = total_profit / n_bets if n_bets else 0.0
    win_rate = bets_df["won"].mean() if n_bets else 0.0

    return {
        "line_edge_threshold": line_edge_threshold,
        "bets": n_bets,
        "profit_units": total_profit,
        "roi": roi,
        "win_rate": win_rate,
        "bets_df": bets_df,
    }


def sweep_line_edge_thresholds(test_df: pd.DataFrame, predicted_totals: np.ndarray):
    thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []

    print("\nTotals regression line-edge sweep")
    print("---------------------------------")

    for threshold in thresholds:
        result = backtest_totals_bets(test_df, predicted_totals, line_edge_threshold=threshold)
        results.append({
            "line_edge_threshold": result["line_edge_threshold"],
            "bets": result["bets"],
            "profit_units": result["profit_units"],
            "roi": result["roi"],
            "win_rate": result["win_rate"],
        })

    results_df = pd.DataFrame(results).sort_values("line_edge_threshold").reset_index(drop=True)
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["roi"].idxmax()]
    best_threshold = float(best_row["line_edge_threshold"])
    best_result = backtest_totals_bets(test_df, predicted_totals, line_edge_threshold=best_threshold)

    print(f"\nBest line-edge threshold: {best_threshold:.1f}")
    print(f"Bets placed: {best_result['bets']}")
    print(f"Profit (units): {best_result['profit_units']:.2f}")
    print(f"ROI: {best_result['roi']:.3%}")
    print(f"Win rate: {best_result['win_rate']:.3%}")

    return results_df, best_result


def train_totals_regression_model(model_df: pd.DataFrame):
    feature_cols = build_feature_columns()

    completed_df = model_df[model_df["is_completed"]].copy()
    completed_df = completed_df.dropna(subset=feature_cols + ["actual_total", "game_date"])

    if len(completed_df) < 20:
        raise ValueError(f"Not enough completed rows to train totals regression model. Found {len(completed_df)} rows.")

    train_part, test_part, split_date = time_based_split(completed_df)

    X_train = train_part[feature_cols]
    y_train = train_part["actual_total"]

    X_test = test_part[feature_cols]
    y_test = test_part["actual_total"]

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=3,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nTotals regression model evaluation")
    print("---------------------------------")
    print(f"Rows used: {len(completed_df)}")
    print(f"Train rows: {len(train_part)}")
    print(f"Test rows: {len(test_part)}")
    print(f"Time split date: {pd.Timestamp(split_date).date()}")
    print(f"MAE:  {mean_absolute_error(y_test, preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")
    print(f"R^2:  {r2_score(y_test, preds):.3f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances")
    print(importances.head(30).to_string())

    threshold_results_df, best_bet_result = sweep_line_edge_thresholds(test_part, preds)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold_results_df": threshold_results_df,
        "best_bet_result": best_bet_result,
        "test_part": test_part,
        "test_preds": preds,
    }


def predict_upcoming_totals(model_df: pd.DataFrame, model, feature_cols, line_edge_threshold: float) -> pd.DataFrame:
    pred_df = model_df[~model_df["is_completed"]].copy()
    pred_df = pred_df.dropna(subset=feature_cols + ["market_total"])

    if pred_df.empty:
        print("\nNo upcoming totals predictions available.")
        return pred_df

    pred_df["predicted_total_points"] = model.predict(pred_df[feature_cols])
    pred_df["line_edge"] = pred_df["predicted_total_points"] - pred_df["market_total"]

    pred_df["predicted_total_side"] = np.where(
        pred_df["predicted_total_points"] >= pred_df["market_total"],
        "Over",
        "Under"
    )

    pred_df["bet_recommendation"] = np.select(
        [
            pred_df["line_edge"] >= line_edge_threshold,
            pred_df["line_edge"] <= -line_edge_threshold,
        ],
        [
            "Bet Over",
            "Bet Under",
        ],
        default="No bet"
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
    totals_market_df = get_market_totals(historical_odds)

    print(f"\nSchedule rows since 2025-12-01: {len(games)}")
    print(f"Completed games: {int(games['is_completed'].sum())}")
    print(f"Upcoming games: {int((~games['is_completed']).sum())}")
    print(f"Team results rows: {len(team_results)}")
    print(f"Historical odds rows: {len(historical_odds)}")
    print(f"Market totals rows: {len(totals_market_df)}")

    team_log = build_team_game_log(games)
    model_df = build_totals_regression_dataset(games, team_results, team_log, totals_market_df)

    model_df = model_df[
        model_df["team_1_adjoe"].notna() &
        model_df["team_2_adjoe"].notna() &
        model_df["market_total"].notna()
    ].copy()

    print(f"\nRows remaining for totals regression model: {len(model_df)}")

    results = train_totals_regression_model(model_df)

    model = results["model"]
    feature_cols = results["feature_cols"]
    threshold_results_df = results["threshold_results_df"]
    best_bet_result = results["best_bet_result"]

    predictions = predict_upcoming_totals(
        model_df,
        model,
        feature_cols,
        line_edge_threshold=float(best_bet_result["line_edge_threshold"]),
    )

    if not threshold_results_df.empty:
        threshold_results_df.to_csv("totals_regression_threshold_sweep_results.csv", index=False)
        print("\nSaved totals_regression_threshold_sweep_results.csv")

    if not best_bet_result["bets_df"].empty:
        best_bet_result["bets_df"].to_csv("totals_regression_backtest_bets.csv", index=False)
        print("Saved totals_regression_backtest_bets.csv")

    if not predictions.empty:
        output_cols = [
            "game_date",
            "team_1",
            "team_2",
            "market_total",
            "predicted_total_points",
            "line_edge",
            "predicted_total_side",
            "bet_recommendation",
        ]
        predictions[output_cols].to_csv("totals_regression_upcoming_predictions.csv", index=False)
        print("\nUpcoming totals regression predictions")
        print(predictions[output_cols].head(25).to_string(index=False))
        print("\nSaved totals_regression_upcoming_predictions.csv")
    else:
        print("\nNo upcoming totals regression predictions generated.")


if __name__ == "__main__":
    main()