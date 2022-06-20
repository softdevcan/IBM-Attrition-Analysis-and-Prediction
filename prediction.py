import joblib
import pandas as pd
import edascripts as eda
import create_df as e

df = eda.load_datasets()
X, y = e.prepare_dataset_fe(df)

random_user = X.sample(1, random_state=17)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)