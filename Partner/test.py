import warnings
from pprint import pprint

import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_validate

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    sgb = XGBRegressor()
    lgb = LGBMRegressor(verbose=0)
    cat = CatBoostRegressor(verbose=0)

    train_data = pd.read_csv('../custom_train_data.csv', index_col='ID')
    test_data = pd.read_csv('../custom_test_data.csv', index_col='ID')
    train_label = pd.read_csv('../custom_train_label.csv', index_col='ID')

    vt = VotingRegressor([('sgb', sgb), ('lgb', lgb), ('cat', cat)])
    res = cross_validate(
        vt, train_data, train_label, scoring='neg_mean_absolute_percentage_error', n_jobs=-1,
        return_train_score=True
    )
    res['test_score_mean'] = res['test_score'].mean() * -100
    res['train_score_mean'] = res['train_score'].mean() * -100
    pprint(res)

    vt.fit(train_data, train_label)
    res = vt.predict(test_data)
    pd.DataFrame({'predicted_price': res}, index=test_data.index).to_csv('public_submission.csv')