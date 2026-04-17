import requests
import pandas as pd
from datetime import datetime, timedelta


def scrape_ncaa_schedule(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """
    Compile NCAA men's Division I basketball schedule data from the NCAA scoreboard-style API.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD, defaults to today

    Returns:
        pandas.DataFrame
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else datetime.today().date()
    )

    if start > end:
        raise ValueError("start_date must be on or before end_date")

    all_games: list[dict] = []
    current = start

    while current <= end:
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        # Mirrors the NCAA scoreboard path for that day
        url = f"https://ncaa-api.henrygd.me/scoreboard/basketball-men/d1/{yyyy}/{mm}/{dd}/all-conf"

        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            print(f"Request failed for {current}: {exc}")
            current += timedelta(days=1)
            continue
        except ValueError:
            print(f"Invalid JSON returned for {current}")
            current += timedelta(days=1)
            continue

        games = data.get("games", [])

        if not games:
            print(f"No games found for {current}")
            current += timedelta(days=1)
            continue

        for item in games:
            game = item.get("game", {})
            home = game.get("home", {})
            away = game.get("away", {})

            all_games.append(
                {
                    "date": game.get("startDate"),
                    "game_id": game.get("gameID"),
                    "title": game.get("title"),
                    "away_team": away.get("names", {}).get("short"),
                    "away_score": away.get("score"),
                    "home_team": home.get("names", {}).get("short"),
                    "home_score": home.get("score"),
                    "status": game.get("gameState"),
                    "current_period": game.get("currentPeriod"),
                    "start_time": game.get("startTime"),
                    "network": game.get("network"),
                    "game_url": game.get("url"),
                }
            )

        print(f"Retrieved {len(games)} games for {current}")
        current += timedelta(days=1)

    df = pd.DataFrame(all_games)

    if not df.empty:
        df.to_excel("ncaa_schedule_compiled.xlsx", index=False)
        print("Saved to ncaa_schedule_compiled.xlsx")
    else:
        print("No schedule data found.")

    return df


# Example: start on November 3, 2025 and go through today
df = scrape_ncaa_schedule("2025-11-03")
print(df.head())