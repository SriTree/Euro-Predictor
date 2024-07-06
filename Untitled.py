#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
import warnings


# In[2]:


#importing csv files
nations_one = pd.read_csv("nations_league_1.csv", index_col=0)
nations_two = pd.read_csv("nations_league_2.csv", index_col=0)
world_cup = pd.read_csv("world_cup.csv", index_col=0)
euro_qual = pd.read_csv("euro_qual.csv", index_col=0)
euro_2022 = pd.read_csv("euro_2022.csv", index_col=0)


# In[3]:


# Function to replace abbreviations with full country names
def remove_abbreviation(opponent):
    return opponent.split(' ', 1)[1]


# In[4]:


#combing all df into one combined df, cleaning up data 
combined = pd.concat([nations_one,nations_two,world_cup,euro_qual,euro_2022])
combined['Opponent'] = combined['Opponent'].apply(remove_abbreviation)
combined = combined[combined['Comp'] != 'Friendlies (M)']
combined.to_csv("matches.csv")


# In[5]:


# Function to adjust rows where the match went to overtime and winner was determined by penalty shoot-out
def adjust_result(row):
    gf = row['GF']
    ga = row['GA']
    
    # Check if there are parentheses in both GF and GA
    if re.search(r'\(\d+\)', gf) and re.search(r'\(\d+\)', ga):
        # Extract the numbers inside the parentheses
        gf_shootout = int(re.search(r'\((\d+)\)', gf).group(1))
        ga_shootout = int(re.search(r'\((\d+)\)', ga).group(1))
        
        # Adjust the result based on shootout scores
        if gf_shootout > ga_shootout:
            return 'W'
        elif gf_shootout < ga_shootout:
            return 'L'
        else:
            return row['Result']  # In case it's still a draw (unlikely scenario)
    else:
        return row['Result']


# In[6]:


combined['GF'] = combined['GF'].astype(str)
combined['GA'] = combined['GA'].astype(str)
combined['Result'] = combined.apply(adjust_result, axis=1)


# In[7]:


# Function to create weighted average for goals for and goals against for matches where winner was determined by penalty shootout
def adjust_goals(goals):
    # Check if there are parentheses indicating shootout scores
    if re.search(r'\(\d+\)', goals):
        # Extract the regular and shootout goals
        regular_goals = int(re.search(r'^\d+', goals).group())
        shootout_goals = int(re.search(r'\((\d+)\)', goals).group(1))
        # Calculate the adjusted goals (average or weighted average)
        adjusted_goals = (regular_goals + shootout_goals) / 2
        return adjusted_goals
    else:
        # Return the regular goals if no shootout score exists
        return float(goals)


# In[8]:


combined['GF'] = combined['GF'].apply(adjust_goals)
combined['GA'] = combined['GA'].apply(adjust_goals)


# In[9]:


combined.shape


# In[10]:


combined.columns = combined.columns.str.lower()
combined = combined.sort_values(by="date")

venue_mapping = {'Home': 1, 'Away': 2, 'Neutral': 3}
combined['venue_num'] = combined['venue'].map(venue_mapping).astype(int)

result_mapping = {'L': 1, 'D': 2, 'W': 3}
combined = combined.dropna(subset=['result'])
combined['target'] = combined['result'].map(result_mapping).astype(int)
combined = combined.dropna(subset=['saves'])
combined['saves'] = combined['saves'].astype(int)
combined = combined.drop(columns=['xg', 'xga'])


# In[11]:


#function to create rolling avg for stats
def rolling_avg(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed = 'left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    return group


# In[12]:


cols = ["gf", "ga", "sh", "sot", "pk", "pkatt", "saves", "cs"]
new_cols = [f"{c}_rolling" for c in cols]

combined_rolling = combined.groupby('nation').apply(lambda x: rolling_avg(x, cols, new_cols))
combined_rolling = combined_rolling.droplevel('nation')
combined_rolling = combined_rolling.sort_values(by="date")


# In[13]:


# Define the features to use for each team
features = ['gf_rolling', 'ga_rolling', 'sh_rolling', 'sot_rolling', 'pk_rolling', 'pkatt_rolling', 'saves_rolling', 'cs_rolling']

# Group by 'nation' and calculate the mean of the features
team_stats_df = combined_rolling.groupby('nation')[features].mean()

# Convert the DataFrame to a dictionary where each key is a nation and each value is a list of features
team_stats = team_stats_df.apply(lambda row: row.tolist(), axis=1).to_dict()


# In[29]:


combined_rolling['target'] = combined_rolling['result']
combined_rolling["venue_code"] = combined_rolling ["venue"].astype("category").cat.codes
combined_rolling["opp_code"] = combined_rolling["opponent"].astype("category").cat.codes
combined_rolling["hour"] = combined_rolling["time"].str.replace(":.+", "", regex=True).astype("int")
combined_rolling["date"] = pd.to_datetime(combined_rolling["date"])
combined_rolling["day_code"] = combined_rolling["date"].dt.dayofweek

new_features = ['venue_code', 'opp_code', 'hour', 'day_code']
combined_rolling.shape


# In[27]:


rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

def make_pred(data, predictors):
    train = combined_rolling[combined_rolling["date"] < '2023-11-17']
    test = combined_rolling[combined_rolling["date"] > '2023-11-17']
    rf.fit(train[predictors], train["target"])
    pred = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predictions=pred), index=test.index)
    prec_score = precision_score(test["target"], pred, average='weighted')
    return combined, prec_score

combined, prec_score = make_pred(combined_rolling, features + new_features)
prec_score


# In[16]:


warnings.filterwarnings('ignore')

# Define the function to simulate the matches
def simulate_match(team1_stats, team2_stats, num_simulations=10):
    results = {'team1_win': 0, 'team2_win': 0, 'draw': 0}
    team1_stats_df = pd.DataFrame([team1_stats], columns=features)
    team2_stats_df = pd.DataFrame([team2_stats], columns=features)
    
    for _ in range(num_simulations):
        team1_score = model.predict(scaler.transform(team1_stats_df))
        team2_score = model.predict(scaler.transform(team2_stats_df))
        if team1_score > team2_score:
            results['team1_win'] += 1
        elif team1_score < team2_score:
            results['team2_win'] += 1
        else:
            results['draw'] += 1
    for key in results:
        results[key] /= num_simulations
    return results

# Define matchups
matchups = [
    ("Germany", "Scotland"),
    ("Hungary", "Switzerland"),
    ("Spain", "Croatia"),
    ("Italy", "Albania"),
    ("Poland", "Netherlands"),
    ("Slovenia", "Denmark"),
    ("Serbia", "England"),
    ("Romania", "Ukraine"),
    ("Belgium", "Slovakia"),
    ("Austria", "France"),
    ("Turkey", "Georgia"),
    ("Portugal", "Czech Republic"),
    ("Croatia", "Albania"),
    ("Germany", "Hungary"),
    ("Scotland", "Switzerland"),
    ("Slovenia", "Serbia"),
    ("Denmark", "England"),
    ("Spain", "Italy"),
    ("Slovakia", "Ukraine"),
    ("Poland", "Austria"),
    ("Netherlands", "France"),
    ("Georgia", "Czech Republic"),
    ("Turkey", "Portugal"),
    ("Belgium", "Romania"),
    ("Switzerland", "Germany"),
    ("Scotland", "Hungary"),
    ("Croatia", "Italy"),
    ("Albania", "Spain"),
    ("Netherlands", "Austria"),
    ("France", "Poland"),
    ("England", "Slovenia"),
    ("Denmark", "Serbia"),
    ("Slovakia", "Romania"),
    ("Ukraine", "Belgium"),
    ("Czech Republic", "Turkey"),
    ("Georgia", "Portugal")
]

# Simulate the matches
for team1, team2 in matchups:
    if team1 in team_stats and team2 in team_stats:
        results = simulate_match(team_stats[team1], team_stats[team2])
        print(f'{team1} vs {team2}: {results}')
    else:
        print(f'Missing stats for {team1} or {team2}')
        

