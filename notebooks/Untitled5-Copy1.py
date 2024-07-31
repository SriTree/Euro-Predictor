#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
import warnings

warnings.filterwarnings('ignore')

# Loading datasets
nations_one = pd.read_csv("nations_league_1.csv", index_col=0)
nations_two = pd.read_csv("nations_league_2.csv", index_col=0)
world_cup = pd.read_csv("world_cup.csv", index_col=0)
euro_qual = pd.read_csv("euro_qual.csv", index_col=0)
euro_2022 = pd.read_csv("euro_2022.csv", index_col=0)

# Function to replace abbreviations with full country names
def remove_abbreviation(opponent):
    return opponent.split(' ', 1)[1]

# Combining all df into one combined df, cleaning up data 
combined = pd.concat([nations_one, nations_two, world_cup, euro_qual, euro_2022])
combined['Opponent'] = combined['Opponent'].apply(remove_abbreviation)
combined = combined[combined['Comp'] != 'Friendlies (M)']
combined.to_csv("matches.csv")

# Function to adjust rows where the match went to overtime and winner was determined by penalty shoot-out
def adjust_result(row):
    gf = row['GF']
    ga = row['GA']
    
    if re.search(r'\(\d+\)', gf) and re.search(r'\(\d+\)', ga):
        gf_shootout = int(re.search(r'\((\d+)\)', gf).group(1))
        ga_shootout = int(re.search(r'\((\d+)\)', ga).group(1))
        
        if gf_shootout > ga_shootout:
            return 'W'
        elif gf_shootout < ga_shootout:
            return 'L'
        else:
            return row['Result']
    else:
        return row['Result']

combined['GF'] = combined['GF'].astype(str)
combined['GA'] = combined['GA'].astype(str)
combined['Result'] = combined.apply(adjust_result, axis=1)

# Function to create weighted average for goals for and goals against for matches where winner was determined by penalty shootout
def adjust_goals(goals):
    if re.search(r'\(\d+\)', goals):
        regular_goals = int(re.search(r'^\d+', goals).group())
        shootout_goals = int(re.search(r'\((\d+)\)', goals).group(1))
        adjusted_goals = (regular_goals + shootout_goals) / 2
        return adjusted_goals
    else:
        return float(goals)

combined['GF'] = combined['GF'].apply(adjust_goals)
combined['GA'] = combined['GA'].apply(adjust_goals)

combined.columns = combined.columns.str.lower()
combined = combined.sort_values(by="date")

venue_mapping = {'Home': 1, 'Away': 2, 'Neutral': 3}
combined['venue_num'] = combined['venue'].map(venue_mapping).astype(int)

# Convert target values to binary (0 for loss, 1 for win)
result_mapping = {'L': 0, 'D': 0, 'W': 1}
combined = combined.dropna(subset=['result'])
combined['target'] = combined['result'].map(result_mapping).astype(int)
combined = combined.dropna(subset=['saves'])
combined['saves'] = combined['saves'].astype(int)
combined = combined.drop(columns=['xg', 'xga'])

# Function to create rolling avg for stats
def rolling_avg(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["gf", "ga", "sh", "sot", "pk", "pkatt", "saves", "cs"]
new_cols = [f"{c}_rolling" for c in cols]

combined_rolling = combined.groupby('nation').apply(lambda x: rolling_avg(x, cols, new_cols))
combined_rolling = combined_rolling.droplevel('nation')
combined_rolling = combined_rolling.sort_values(by="date")

# Adding additional feature columns
combined_rolling["venue_code"] = combined_rolling["venue"].astype("category").cat.codes
combined_rolling["opp_code"] = combined_rolling["opponent"].astype("category").cat.codes
combined_rolling["hour"] = combined_rolling["time"].str.replace(":.+", "", regex=True).astype(int)
combined_rolling["date"] = pd.to_datetime(combined_rolling["date"])
combined_rolling["day_code"] = combined_rolling["date"].dt.dayofweek

# Define the features to use for each team
features = ['gf_rolling', 'ga_rolling', 'sh_rolling', 'sot_rolling', 'pk_rolling', 'pkatt_rolling', 'saves_rolling', 'cs_rolling',
            'venue_code', 'opp_code', 'hour', 'day_code']


# In[2]:


import pandas as pd
import re
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
import warnings

warnings.filterwarnings('ignore')

# Assuming 'combined_rolling' is already prepared as per your previous steps

# Features for model training
features = ['gf_rolling', 'ga_rolling', 'sh_rolling', 'sot_rolling', 'pk_rolling', 'pkatt_rolling', 
            'saves_rolling', 'cs_rolling', 'venue_code', 'opp_code', 'hour', 'day_code']

# Define target variable
target = 'target'

# Split into training and testing sets (e.g., 80% training, 20% testing)
msk = np.random.rand(len(combined_rolling)) < 0.8
train_df = combined_rolling[msk]
test_df = combined_rolling[~msk]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


# In[3]:


# Convert datasets to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# Define model parameters
param = {
    'verbosity': 1, 
    'objective': 'binary:logistic',  # Use logistic regression for binary classification
    'eval_metric': 'logloss',  # Log loss evaluation metric
    'booster': 'gbtree',  # Use tree-based models
    'learning_rate': 0.1,
    'max_depth': 5,
    'lambda': 1,  # L2 regularization term
    'alpha': 0  # L1 regularization term
}

# Specify number of boosting rounds
num_round = 100

# Train the model
bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')])

# Predict on test set
y_pred = bst.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')


# In[4]:


import pandas as pd
import re
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score
import warnings

warnings.filterwarnings('ignore')

# Assuming 'combined_rolling' is already prepared as per your previous steps

# Define columns and features
cols = ["gf", "ga", "sh", "sot", "pk", "pkatt", "saves", "cs"]
rolling_avg_cols = ['gf_rolling', 'ga_rolling', 'sh_rolling', 'sot_rolling', 'pk_rolling', 'pkatt_rolling', 'saves_rolling', 'cs_rolling']
features = rolling_avg_cols + ['venue_code', 'opp_code', 'hour', 'day_code']

# Function to compute rolling averages for a team
def compute_rolling_avg(df, team_col, date_col, cols, new_cols):
    df = df.sort_values(by=date_col)
    rolling_df = df.groupby(team_col)[cols].rolling(window=3, min_periods=1, closed='left').mean().reset_index(level=0, drop=True)
    rolling_df.columns = new_cols
    return rolling_df

# Ensure combined_rolling has unique index
combined_rolling = combined_rolling.reset_index(drop=True)

# Calculate rolling averages for the combined_rolling dataframe
combined_rolling[rolling_avg_cols] = compute_rolling_avg(combined_rolling, 'nation', 'date', cols, rolling_avg_cols)

# Function to determine group standings based on match results
def determine_group_standings(results):
    groups = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': []}
    standings = {group: {} for group in groups.keys()}
    
    for result in results:
        group = result['group']
        match = result['match']
        winner = result['winner']
        nation, opponent = match.split(' vs ')
        
        if winner not in standings[group]:
            standings[group][winner] = {'points': 0, 'goals_for': 0, 'goals_against': 0}
        if opponent not in standings[group]:
            standings[group][opponent] = {'points': 0, 'goals_for': 0, 'goals_against': 0}
        
        standings[group][winner]['points'] += 3
        standings[group][winner]['goals_for'] += 1
        standings[group][opponent]['goals_against'] += 1
    
    for group in standings:
        sorted_standings = sorted(standings[group].items(), key=lambda item: (-item[1]['points'], item[1]['goals_for'] - item[1]['goals_against']))
        groups[group] = [(team, stats) for team, stats in sorted_standings]
    
    return groups

# Define the group stage matches with groups included
group_stage_matches = {
    'date': [
        '2024-06-14', '2024-06-15', '2024-06-15', '2024-06-15', '2024-06-16', '2024-06-16', '2024-06-16',
        '2024-06-17', '2024-06-17', '2024-06-17', '2024-06-18', '2024-06-18', '2024-06-19', '2024-06-19', '2024-06-19',
        '2024-06-20', '2024-06-20', '2024-06-20', '2024-06-21', '2024-06-21', '2024-06-21', '2024-06-22', '2024-06-22', '2024-06-22', 
        '2024-06-23', '2024-06-23', '2024-06-24', '2024-06-24', '2024-06-25', '2024-06-25', '2024-06-25', '2024-06-25', '2024-06-26', 
        '2024-06-26', '2024-06-26', '2024-06-26'
    ],
    'group': [
        'A', 'A', 'B', 'B', 'D', 'C', 'C', 'E', 'E', 'D', 'F', 'F', 'B', 'A', 'A', 'C', 'C', 'B', 'E', 'D', 
        'D', 'F', 'F', 'E', 'A', 'A', 'B', 'B', 'D', 'D', 'C', 'C', 'E', 'E', 'F', 'F'
    ],
    'nation': [
        'Germany', 'Hungary', 'Spain', 'Italy', 'Poland', 'Slovenia', 'Serbia', 'Romania', 'Belgium', 'Austria',
        'Turkey', 'Portugal', 'Croatia', 'Germany', 'Scotland', 'Slovenia', 'Denmark', 'Spain', 'Slovakia', 'Poland',
        'Netherlands', 'Georgia', 'Turkey', 'Belgium', 'Switzerland', 'Scotland', 'Albania', 'Croatia', 'France', 'Netherlands', 
        'England', 'Denmark', 'Ukraine', 'Slovakia', 'Czechia', 'Georgia'
    ],
    'opponent': [
        'Scotland', 'Switzerland', 'Croatia', 'Albania', 'Netherlands', 'Denmark', 'England', 'Ukraine', 'Slovakia', 'France',
        'Georgia', 'Czechia', 'Albania', 'Hungary', 'Switzerland', 'Serbia', 'England', 'Italy', 'Ukraine', 'Austria', 
        'France', 'Czechia', 'Portugal', 'Romania', 'Germany', 'Hungary', 'Spain', 'Italy', 'Poland', 'Austria', 
        'Slovenia', 'Serbia', 'Belgium', 'Romania', 'Turkey', 'Portugal'
    ],
    'venue': [
        'Home', 'Neutral', 'Home', 'Home', 'Home', 'Neutral', 'Neutral', 'Home', 'Home', 'Home', 
        'Home', 'Home', 'Neutral', 'Home', 'Neutral', 'Home', 'Neutral', 'Home', 'Home', 'Neutral', 
        'Neutral', 'Home', 'Home', 'Neutral', 'Neutral', 'Home', 'Neutral', 'Neutral', 'Home', 'Neutral',
        'Home', 'Neutral', 'Neutral', 'Neutral', 'Home', 'Home'
    ],
    'time': [
        '21:00', '15:00', '18:00', '21:00', '18:00', '15:00', '21:00', '18:00', '21:00', '21:00', 
        '18:00', '21:00', '18:00', '21:00', '18:00', '18:00', '21:00', '21:00', '18:00', '21:00', 
        '21:00', '15:00', '18:00', '21:00', '21:00', '21:00', '21:00', '21:00', '18:00', '18:00',
        '21:00', '21:00', '18:00', '18:00', '21:00', '21:00'
    ]
}

group_stage_df = pd.DataFrame(group_stage_matches)

# Add feature columns
group_stage_df['date'] = pd.to_datetime(group_stage_df['date'])
group_stage_df['venue_code'] = group_stage_df.apply(lambda row: 1 if row['nation'] == 'Germany' else 2 if row['opponent'] == 'Germany' else 3, axis=1)
group_stage_df['opp_code'] = group_stage_df['opponent'].astype('category').cat.codes
group_stage_df['hour'] = group_stage_df['time'].str.replace(':.+', '', regex=True).astype(int)
group_stage_df['day_code'] = group_stage_df['date'].dt.dayofweek

# Merge rolling averages into group_stage_df
group_stage_df = group_stage_df.merge(combined_rolling[['nation'] + rolling_avg_cols].drop_duplicates(subset='nation'), how='left', on='nation')

# Fill missing rolling averages with the mean values from combined_rolling
for col in rolling_avg_cols:
    if col not in group_stage_df.columns:
        group_stage_df[col] = combined_rolling[col].mean()

# Function to predict match outcomes
def predict_match_outcomes(match_data, model, features):
    match_dmatrix = xgb.DMatrix(match_data[features])
    predictions = model.predict(match_dmatrix)
    return predictions

# Predict outcomes
group_stage_df['prediction'] = predict_match_outcomes(group_stage_df, bst, features)

# Function to display match results and determine winners
def display_match_results_and_determine_winners(df, stage_name):
    results = []
    print(f"\n{stage_name} Matches:")
    for index, row in df.iterrows():
        win_prob = row['prediction']
        nation = row['nation']
        opponent = row['opponent']
        winner = nation if win_prob > 0.5 else opponent
        results.append({'group': row['group'], 'match': f'{nation} vs {opponent}', 'winner': winner})
        print(f'The winner of {nation} vs {opponent} is predicted to be {winner}')
    return results

# Display the group stage match results and get the results
group_stage_results = display_match_results_and_determine_winners(group_stage_df, "Group Stage")

# Determine group standings
group_standings = determine_group_standings(group_stage_results)

# Determine top 2 teams and best 4 third-placed teams for the Round of 16
top_2_teams = {group: standings[:2] for group, standings in group_standings.items()}

# Collect third-placed teams and their stats
third_placed_teams = []
for group, standings in group_standings.items():
    if len(standings) > 2:
        third_placed_team = standings[2]
        third_placed_teams.append({'team': third_placed_team[0], 'stats': third_placed_team[1]})

# Debugging: Print the third_placed_teams to verify structure
print("\nThird-placed teams and their stats:")
for team in third_placed_teams:
    print(team)

# Sort third-placed teams based on points and goal difference
third_placed_teams.sort(key=lambda x: (x['stats']['points'], x['stats']['goals_for'] - x['stats']['goals_against']), reverse=True)

# Get the best 4 third-placed teams
best_4_third_placed = [team['team'] for team in third_placed_teams[:4]]

# Print advancing teams
print(f"\nTop 2 teams from each group: {top_2_teams}")
print(f"Best 4 third-placed teams: {best_4_third_placed}")

# Function to simulate the knockout rounds
def simulate_knockout_rounds(top_2_teams, best_4_third_placed):
    # Round of 16
    print("\nRound of 16 Matches:")
    round_of_16_matches = [
        (top_2_teams['A'][0][0], best_4_third_placed[3]),
        (top_2_teams['B'][0][0], top_2_teams['A'][1][0]),
        (top_2_teams['C'][0][0], best_4_third_placed[1]),
        (top_2_teams['D'][0][0], top_2_teams['E'][1][0]),
        (top_2_teams['E'][0][0], top_2_teams['D'][1][0]),
        (top_2_teams['F'][0][0], best_4_third_placed[2]),
        (top_2_teams['C'][1][0], best_4_third_placed[0]),
        (top_2_teams['B'][1][0], top_2_teams['F'][1][0])
    ]
    
    round_of_16_results = []
    for match in round_of_16_matches:
        match_df = pd.DataFrame([{
            'nation': match[0],
            'opponent': match[1],
            'venue_code': 1 if match[0] == 'Germany' else 2 if match[1] == 'Germany' else 3,
            'opp_code': 1 if match[1] == 'Germany' else 2 if match[0] == 'Germany' else 3,
            'hour': 21,
            'day_code': 6,
        }])
        match_df = match_df.merge(combined_rolling[['nation'] + rolling_avg_cols].drop_duplicates(subset='nation'), how='left', on='nation')
        for col in rolling_avg_cols:
            if col not in match_df.columns:
                match_df[col] = combined_rolling[col].mean()
        match_df['prediction'] = predict_match_outcomes(match_df, bst, features)
        win_prob = match_df['prediction'].values[0]
        winner = match[0] if win_prob > 0.5 else match[1]
        round_of_16_results.append(winner)
        print(f'The winner of {match[0]} vs {match[1]} is predicted to be {winner}')
    
    # Quarterfinals
    print("\nQuarterfinal Matches:")
    quarterfinal_matches = [
        (round_of_16_results[0], round_of_16_results[1]),
        (round_of_16_results[2], round_of_16_results[3]),
        (round_of_16_results[4], round_of_16_results[5]),
        (round_of_16_results[6], round_of_16_results[7])
    ]
    
    quarterfinal_results = []
    for match in quarterfinal_matches:
        match_df = pd.DataFrame([{
            'nation': match[0],
            'opponent': match[1],
            'venue_code': 1 if match[0] == 'Germany' else 2 if match[1] == 'Germany' else 3,
            'opp_code': 1 if match[1] == 'Germany' else 2 if match[0] == 'Germany' else 3,
            'hour': 21,
            'day_code': 6,
        }])
        match_df = match_df.merge(combined_rolling[['nation'] + rolling_avg_cols].drop_duplicates(subset='nation'), how='left', on='nation')
        for col in rolling_avg_cols:
            if col not in match_df.columns:
                match_df[col] = combined_rolling[col].mean()
        match_df['prediction'] = predict_match_outcomes(match_df, bst, features)
        win_prob = match_df['prediction'].values[0]
        winner = match[0] if win_prob > 0.5 else match[1]
        quarterfinal_results.append(winner)
        print(f'The winner of {match[0]} vs {match[1]} is predicted to be {winner}')
    
    # Semifinals
    print("\nSemifinal Matches:")
    semifinal_matches = [
        (quarterfinal_results[0], quarterfinal_results[1]),
        (quarterfinal_results[2], quarterfinal_results[3])
    ]
    
    semifinal_results = []
    for match in semifinal_matches:
        match_df = pd.DataFrame([{
            'nation': match[0],
            'opponent': match[1],
            'venue_code': 1 if match[0] == 'Germany' else 2 if match[1] == 'Germany' else 3,
            'opp_code': 1 if match[1] == 'Germany' else 2 if match[0] == 'Germany' else 3,
            'hour': 21,
            'day_code': 6,
        }])
        match_df = match_df.merge(combined_rolling[['nation'] + rolling_avg_cols].drop_duplicates(subset='nation'), how='left', on='nation')
        for col in rolling_avg_cols:
            if col not in match_df.columns:
                match_df[col] = combined_rolling[col].mean()
        match_df['prediction'] = predict_match_outcomes(match_df, bst, features)
        win_prob = match_df['prediction'].values[0]
        winner = match[0] if win_prob > 0.5 else match[1]
        semifinal_results.append(winner)
        print(f'The winner of {match[0]} vs {match[1]} is predicted to be {winner}')
    
    # Final
    print("\nFinal Match:")
    final_match = (semifinal_results[0], semifinal_results[1])
    match_df = pd.DataFrame([{
        'nation': final_match[0],
        'opponent': final_match[1],
        'venue_code': 1 if final_match[0] == 'Germany' else 2 if final_match[1] == 'Germany' else 3,
        'opp_code': 1 if final_match[1] == 'Germany' else 2 if final_match[0] == 'Germany' else 3,
        'hour': 21,
        'day_code': 6,
    }])
    match_df = match_df.merge(combined_rolling[['nation'] + rolling_avg_cols].drop_duplicates(subset='nation'), how='left', on='nation')
    for col in rolling_avg_cols:
        if col not in match_df.columns:
            match_df[col] = combined_rolling[col].mean()
    match_df['prediction'] = predict_match_outcomes(match_df, bst, features)
    win_prob = match_df['prediction'].values[0]
    winner = final_match[0] if win_prob > 0.5 else final_match[1]
    print(f'The winner of the final match {final_match[0]} vs {final_match[1]} is predicted to be {winner}')
    return winner

# Simulate the knockout rounds and determine the tournament winner
tournament_winner = simulate_knockout_rounds(top_2_teams, best_4_third_placed)
print(f'\nThe predicted winner of the Euro tournament is {tournament_winner}')

