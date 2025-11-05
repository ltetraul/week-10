import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#load dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
data = pd.read_csv(url)

#prepare roast mapping
roast_categories = data['roast'].dropna().unique()
roast_map = {category: idx + 1 for idx, category in enumerate(roast_categories)}
data['roast_num'] = data['roast'].map(roast_map)

#model 1
X1 = data[['100g_USD']]
y1 = data['rating']

model_1 = LinearRegression()
model_1.fit(X1, y1)

#model 2
X2 = data[['100g_USD', 'roast_num']].dropna()
y2 = data.loc[X2.index, 'rating']

model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y2)

#save both models and roast map
with open("model_1.pickle", "wb") as f:
    pickle.dump(model_1, f)

with open("model_2.pickle", "wb") as f:
    pickle.dump(model_2, f)

with open("roast_map.pickle", "wb") as f:
    pickle.dump(roast_map, f)

print("✅ model_1 (Linear Regression) saved.")
print("✅ model_2 (Decision Tree Regressor) saved.")
print("✅ roast_map saved.")