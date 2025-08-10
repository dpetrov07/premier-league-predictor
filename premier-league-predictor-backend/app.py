from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Reads in match data and parses through non numerical values
def get_matches_data():
    matches = pd.read_csv('/content/drive/My Drive/matches.csv', index_col=0)
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent "].astype("category").cat.codes
    matches["result_code"] = matches["result"].map({"W": 1, "D": 0, "L": -1})
    pd.set_option('display.max_columns', None)
    return matches

# Computes averages for a teams previous three matches
def get_three_match_rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats

    # Removes matches without enough previous data
    group = group.dropna(subset=new_cols)
    return group

# Passes in specific data to retrieve rolling averages
def get_rolling_averages(matches):
    cols = ["gf", "ga", "sh", "sot", "xg", "xga"]
    new_cols = [f"{c}_rolling" for c in cols]
    # Splits table by each teams data and fixes indexs
    matches_rolling = matches.groupby("team").apply(
        lambda x: get_three_match_rolling_averages(x, cols, new_cols)
    ).reset_index(level=0, drop=True)
    matches_rolling.index = range(matches_rolling.shape[0])
    return matches_rolling

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
mapping = MissingDict(**map_values)

# Trains data with RFC and returns accuracy & precision
def use_random_forest_classifier(data, predictors):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] >= '2022-01-01']

    # Trains model and predicts future matches
    rf.fit(train[predictors], train["result_code"])
    preds = rf.predict(test[predictors])


    combined = pd.DataFrame({
        "actual": test["result_code"],
        "predicted": preds,
        "team": test["team"].map(mapping),
        "opponent": test["opponent"].map(mapping)
    }, index=test.index)

    precision = precision_score(test["result_code"], preds, average='macro')
    return combined, precision


#
# Main method
# 
matches = get_matches_data()
matches_rolling = get_rolling_averages(matches)

# Adds calculated rolling averages stats into table
matches = matches.merge(
    matches_rolling[["date", "team"] + [col for col in matches_rolling.columns if col.endswith('_rolling')]],
    on=["date", "team"],
    how="left"
)

predictors = ["venue_code", "opp_code"] + [c for c in matches.columns if c.endswith("_rolling")]
combined, precision = use_random_forest_classifier(matches, predictors)

# Adds model predictions into table with rest of data
combined = combined.merge(
    matches_rolling[["date", "team", "opponent", "result"]],
    left_index=True,
    right_index=True
)

print(f"Precision: {precision}")
display(combined.sort_values("date", ascending=True))
