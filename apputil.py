import os
import pickle
import pandas as pd
import numpy as np

#locate the model folder
home_dir = os.path.expanduser("~")
model_dir = os.path.join(home_dir, "week10_models")

#load models and roast map
model_1_path = os.path.join(model_dir, "model_1.pickle")
model_2_path = os.path.join(model_dir, "model_2.pickle")
roast_map_path = os.path.join(model_dir, "roast_map.pickle")

with open(model_1_path, "rb") as f:
    model_1 = pickle.load(f)

with open(model_2_path, "rb") as f:
    model_2 = pickle.load(f)

with open(roast_map_path, "rb") as f:
    roast_map = pickle.load(f)

#define prediction function
def predict_rating(df_X: pd.DataFrame):
    """
    Takes a DataFrame with columns:
        - 100g_USD (float)
        - roast (str)
    Returns:
        - numpy array of predicted ratings
    """

    #input validation
    if not all(col in df_X.columns for col in ['100g_USD', 'roast']):
        raise ValueError("DataFrame must contain '100g_USD' and 'roast' columns")

    predictions = []

    for _, row in df_X.iterrows():
        usd = row['100g_USD']
        roast = row['roast']

        #use model 2 if roast is known
        if roast in roast_map:
            roast_num = roast_map[roast]
            X_input = pd.DataFrame([[usd, roast_num]], columns=['100g_USD', 'roast_num'])
            pred = model_2.predict(X_input)[0]
        else:
            #use model 1 if roast is unknown
            X_input = pd.DataFrame([[usd]], columns=['100g_USD'])
            pred = model_1.predict(X_input)[0]

        predictions.append(pred)

    return np.array(predictions)