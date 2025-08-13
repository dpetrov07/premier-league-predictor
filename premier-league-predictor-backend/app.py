import uuid
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# Normalizes different team name data
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
team_name_normalizer = MissingDict(**map_values)


# Reads in match data and parses through non numerical values
def get_matches_data():
    matches = pd.read_csv("matches.csv", index_col=0)
    matches["team"] = matches["team"].map(team_name_normalizer)
    matches["opponent"] = matches["opponent"].map(team_name_normalizer)
    matches["date"] = pd.to_datetime(matches["date"])
    matches["id"] = matches.apply(lambda row: uuid.uuid4(), axis=1)
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["result_code"] = matches["result"].map({"W": 1, "D": 0, "L": -1})
    pd.set_option("display.max_columns", None)
    return matches

# Gets unique future matches so model only predicts each match once
def get_unique_future_matches(matches=None):
    if matches is None:    
        matches = get_matches_data()
    
    future_matches = matches[matches["date"] > "2022-01-01"].sort_values(by=["date", "time"])
    
    future_matches["match_teams"] = future_matches.apply(
        lambda row: tuple(sorted([row["team"], row["opponent"]])), axis=1
    )
    future_matches = future_matches.drop_duplicates(subset=["date", "match_teams"], keep="first")
    return future_matches

# Computes rolling averages for a team's previous matches
def get_rolling_averages(matches, window=3, team_col="team"):
    cols = ["gf", "ga", "sh", "sot", "xg", "xga"]
    new_cols = [f"{c}_rolling{window}" for c in cols]

    # Computes average scores for cols in team's previous matches
    def compute_team_rolling(group):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(window, closed="left").mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group

    # Apply to each team separately
    matches_rolling = matches.groupby(team_col).apply(compute_team_rolling, include_groups=False)
    matches_rolling = matches_rolling.reset_index(level=0, drop=True)
    matches_rolling.index = range(matches_rolling.shape[0])

    return matches_rolling
  
# Computes rolling averages for opponent's previous matches
def get_opponent_rolling_averages(matches, window=3):
    # Reuses get_rolling_averages function
    opposition_rolling = get_rolling_averages(matches, window=window, team_col="opponent")
    
    #Rename opposition rolling columns to differentiate
    cols = ["gf", "ga", "sh", "sot", "xg", "xga"]
    for c in cols:
        old_col = f"{c}_rolling{window}"
        new_col = f"opp_{c}_rolling{window}"
        opposition_rolling.rename(columns={old_col: new_col}, inplace=True)
        
    return opposition_rolling

# Trains data with RFC and returns accuracy & precision
def use_random_forest_classifier(data, predictors):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    train = data[data["date"] < "2022-01-01"]
    test = data[data["date"] > "2022-01-01"]

    # Trains model and predicts future matches after 2022-01-01
    rf.fit(train[predictors], train["result_code"])
    preds = rf.predict(test[predictors])

    # Creates table for displaying with actual data and predicted data
    combined_data = pd.DataFrame({
        "date": test["date"],
        "actual": test["result_code"],
        "predicted": preds,
        "team": test["team"],
        "opponent": test["opponent"],
    }, index=test.index)

    precision = precision_score(test["result_code"], preds, average="macro")
    
    combined_data = combined_data.sort_values("date").reset_index(drop=True)
    
    return combined_data, precision


#
# API calls
#
# app = Flask(__name__)
# CORS(app)

# @app.route("/api/future-matches")
# def get_future_matches():
#     matches = get_matches_data()
#     # Gets matches after Jan 1 2022
#     future_matches = matches[matches["date"] > "2022-01-01"].sort_values(by=["date", "time"])

#     # Removes duplicate matches based on date and new column of combined teams
#     future_matches["match_teams"] = future_matches.apply(lambda row: tuple(sorted([row["team"], 
#         row["opponent"]])), axis=1)
#     future_matches = future_matches.drop_duplicates(subset=["date", "match_teams"], keep="first")
    
#     # Gets next 10 matches and formats into readable date
#     future_ten_matches = future_matches[["date", "time", "team", "opponent"]].head(10)
#     future_ten_matches["date"] = future_ten_matches["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
#     return jsonify(future_ten_matches.to_dict(orient="records"))

# if __name__ == "__main__":
#     app.run(debug=True, port=5050)

# @app.route("/api/predict-match", methods=["POST"])
# def get_match_prediction():
#     return


#
# Main method
# 
matches_data = get_matches_data()
matches_rolling_3 = get_rolling_averages(matches_data, window=3)
opposition_rolling_3 = get_opponent_rolling_averages(matches_data, window=3)

# Merge opponent rolling data into team rolling data
matches_combined_rolling = matches_rolling_3.merge(
    # Compares team and opponent in both datasets and merges same data
    opposition_rolling_3[
        ["team", "date"] +
        [col for col in opposition_rolling_3.columns if col.startswith("opp_")]
    ],
    left_on=["opponent", "date"],
    right_on=["team", "date"],
    how="left"
)
matches_combined_rolling = matches_combined_rolling.rename(columns={
    "opp_code_x": "opp_code",
    "team_x": "team"
})
matches_combined_rolling = matches_combined_rolling.drop(columns=["opp_code_y"])

model_predictors = [
    "venue_code", "opp_code",
    "gf_rolling3", "ga_rolling3",
    "sh_rolling3", "sot_rolling3",
    "xg_rolling3", "xga_rolling3",
    "opp_gf_rolling3", "opp_ga_rolling3",
    "opp_sh_rolling3", "opp_sot_rolling3",
    "opp_xg_rolling3", "opp_xga_rolling3"
]

model_data, precision = use_random_forest_classifier(matches_combined_rolling, model_predictors)

print("Model precision: ", precision)
model_data.to_csv("future_match_predictions.csv", index=False)