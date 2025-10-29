import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle

#load the coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

#prepare the data for training
roast_categories = data['roast'].dropna().unique()
roast_map = {category: idx + 1 for idx, category in enumerate(roast_categories)}  # start from 1
data['roast_num'] = data['roast'].map(roast_map)

#define variables for model training
X = data[['100g_USD', 'roast_num']]
y = data['rating']

#train decision tree regressor model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

#choose a writable folder
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "week10_models")
os.makedirs(model_dir, exist_ok=True)

#save the trained model
model_path = os.path.join(model_dir, "model_2.pickle")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

#save roast category mapping dictionary
map_path = os.path.join(model_dir, "roast_map.pickle")
with open(map_path, "wb") as f:
    pickle.dump(roast_map, f)

print(f"✅ Decision Tree model trained and saved at:\n{model_path}")
print(f"✅ Roast category map saved at:\n{map_path}")
