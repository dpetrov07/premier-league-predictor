import uuid
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

app = Flask(__name__)
CORS(app)

matches_data = pd.read_csv("matches.csv", index_col=0)

# Normalizes different team name data
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}

# Reads in match data and parses through non numerical values
matches_data["team"] = matches_data["team"].map(lambda x: map_values.get(x, x))
matches_data["opponent"] = matches_data["opponent"].map(lambda x: map_values.get(x, x))
matches_data["date"] = pd.to_datetime(matches_data["date"])
matches_data["id"] = [str(uuid.uuid4()) for _ in range(len(matches_data))]
matches_data["venue_code"] = matches_data["venue"].astype("category").cat.codes
matches_data["opp_code"] = matches_data["opponent"].astype("category").cat.codes
matches_data["result_code"] = matches_data["result"].map({"W": 1, "D": 0, "L": -1})

#
# Functions
#

# Gets unique matches so model only predicts each match once
def get_unique_matches(matches):
    
    matches["match_teams"] = matches.apply(lambda row: (row["team"], row["opponent"]), axis=1)
    matches = matches.drop_duplicates(subset=["date", "match_teams"], keep="first")
    return matches

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
    
    matches_rolling = (
        matches.groupby(team_col, group_keys=False)
        .apply(lambda g: compute_team_rolling(g.copy()))
        .reset_index(drop=True)
    )

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
def use_random_forest_classifier(train_data, prediction_data, predictors):
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=12, random_state=3)
    
    # Trains model and predicts future matches after 2022-01-01
    rf.fit(train_data[predictors], train_data["result_code"])
    probs = rf.predict_proba(prediction_data[predictors])
    
    # Map probabilites into unique columns
    prob_map = {-1: "away_win_prob", 0: "draw_prob", 1: "home_win_prob"}
    for i, cls in enumerate(rf.classes_):
        prediction_data[prob_map[cls]] = probs[:, i]
    
    # Creates most likely column predictions and actual result
    prediction_data["predicted"] = [ {-1:"L",0:"D",1:"W"}[cls] for cls in rf.classes_[probs.argmax(axis=1)] ]
    prediction_data["actual"] = prediction_data["result_code"].map({-1: "L", 0: "D", 1: "W"})
    prediction_data = prediction_data.sort_values("date").reset_index(drop=True)
    
    precision = precision_score(prediction_data["actual"], prediction_data["predicted"], average="macro")
    
    return prediction_data, precision

# Combines necessary data to pass into model to predict future matches
def get_future_match_predictions(matches_data, window=3):
    # Compute rolling averages
    matches_rolling = get_rolling_averages(matches_data, window=window)
    opposition_rolling = get_opponent_rolling_averages(matches_data, window=window)

    model_predictors = ["venue_code", "opp_code", "gf_rolling3", "ga_rolling3", "xg_rolling3", 
                    "xga_rolling3", "opp_gf_rolling3", "opp_ga_rolling3", "opp_xg_rolling3", "opp_xga_rolling3"]

    # Merge team and opponent rolling stats
    combined_data = matches_rolling.merge(
        # Compares team and opponent in both datasets and merges same data
        opposition_rolling[
            ["team", "date", "opp_code"] +
            [col for col in opposition_rolling.columns if col.startswith("opp_")]
        ],
        left_on=["opponent", "date"],
        right_on=["team", "date"],
        how="left"
    )

    # Drops duplicate columns and renames properly
    combined_data = combined_data.drop(columns=["team_y", "opp_code_y"]).rename(
        columns={"team_x": "team", "opp_code_x": "opp_code"}
    )
    
    train_data = combined_data[combined_data["date"] < "2022-01-01"]
    prediction_data = get_unique_matches(combined_data[combined_data["date"] > "2022-01-01"])

    predictions, precision = use_random_forest_classifier(train_data, prediction_data, model_predictors)
    return predictions, precision

#
# API calls
#

# Returns next 10 future matches
@app.route("/api/future-matches")
def get_future_matches():
    # Default page data for displaying matches
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 10))
    
    # Gets matches after Jan 1 2022
    future_matches = matches_data[matches_data["date"] > "2022-01-01"].sort_values(by=["date", "time"])

    # Removes duplicate matches based on date and new column of combined teams
    future_matches["match_teams"] = future_matches.apply(lambda row: tuple(sorted([row["team"], 
        row["opponent"]])), axis=1)
    future_matches = future_matches.drop_duplicates(subset=["date", "match_teams"], keep="first")
    
    # Gets next 10 matches and formats into readable date
    future_matches = future_matches[["id", "date", "time", "team", "opponent"]]
    future_matches["date"] = future_matches["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
    
    # Gets matches for requested page
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_matches = future_matches.iloc[start_idx:end_idx]
    
    return jsonify(page_matches.to_dict(orient="records"))

# Returns probabilities for win, tie, and loss percentages
@app.route("/api/predict-match", methods=["POST"])
def get_match_prediction():
    data = request.get_json()
    match_id = data.get("matchId")

    predictions, _ = get_future_match_predictions(matches_data)
    match_prediction = predictions[predictions["id"].astype(str) == match_id]
    match_probs = match_prediction[["home_win_prob", "draw_prob", "away_win_prob"]].iloc[0] * 100
    
    return jsonify(match_probs.round(1).to_dict())

if __name__ == "__main__":
    app.run(debug=True, port=5050)
