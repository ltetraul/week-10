import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle

#load coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

#prepare data for training
roast_categories = data['roast'].dropna().unique()
roast_map = {category: idx + 1 for idx, category in enumerate(roast_categories)}  # start from 1
data['roast_num'] = data['roast'].map(roast_map)

#define variables
X_full = data[['100g_USD', 'roast_num']].dropna()
y_full = data.loc[X_full.index, 'rating']

X_simple = data[['100g_USD']]
y_simple = data['rating']

#train decision tree models
model_1 = DecisionTreeRegressor(random_state=42)  # Uses only 100g_USD
model_1.fit(X_simple, y_simple)

model_2 = DecisionTreeRegressor(random_state=42)  # Uses 100g_USD + roast_num
model_2.fit(X_full, y_full)

#choose folder
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "week10_models")
os.makedirs(model_dir, exist_ok=True)

#save both models
model_1_path = os.path.join(model_dir, "model_1.pickle")
with open(model_1_path, "wb") as f:
    pickle.dump(model_1, f)

model_2_path = os.path.join(model_dir, "model_2.pickle")
with open(model_2_path, "wb") as f:
    pickle.dump(model_2, f)

#roast category mapping dictionary
map_path = os.path.join(model_dir, "roast_map.pickle")
with open(map_path, "wb") as f:
    pickle.dump(roast_map, f)

print(f"✅ Model 1 (100g_USD only) saved at: {model_1_path}")
print(f"✅ Model 2 (100g_USD + roast_num) saved at: {model_2_path}")
print(f"✅ Roast map saved at: {map_path}")

