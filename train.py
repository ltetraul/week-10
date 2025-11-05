import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor

#load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

#map roast categories to numbers
roast_categories = data['roast'].dropna().unique()
roast_map = {category: idx + 1 for idx, category in enumerate(roast_categories)}
data['roast_num'] = data['roast'].map(roast_map)

#model 1
X1 = data[['100g_USD']]
y1 = data['rating']

#model 2
X2 = data[['100g_USD', 'roast_num']].dropna()
y2 = data.loc[X2.index, 'rating']

#train both models
model_1 = DecisionTreeRegressor(random_state=42)
model_1.fit(X1, y1)

model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y2)

#save models to current directory
with open("model_1.pickle", "wb") as f:
    pickle.dump(model_1, f)

with open("model_2.pickle", "wb") as f:
    pickle.dump(model_2, f)

#save roast_map
with open("roast_map.pickle", "wb") as f:
    pickle.dump(roast_map, f)

print("model_1.pickle, model_2.pickle, and roast_map.pickle created successfully.")
