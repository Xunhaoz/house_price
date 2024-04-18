import warnings

from pprint import pprint
import pandas as pd
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_data = pd.read_csv('clean_data_minmax_.csv').drop(columns=['單價'])
    train_label = pd.read_csv('clean_data_minmax_.csv')[['單價']]

    model = XGBRegressor()
    res = cross_validate(model, train_data, train_label, scoring='neg_mean_absolute_percentage_error',
                         return_train_score=True, verbose=1)
    pprint(res)
