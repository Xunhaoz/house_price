import warnings

from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings('ignore')


def randoforest_regressor_mape(train_data, train_label):
    # base random_forest_regressor
    params = [
        {
            'n_estimators': 370,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 440,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 470,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 510,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        }
    ]

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,
                                                        test_size=0.33, random_state=42)

    pred = []
    oob_score_ = 0
    for param in params:
        # print('\n', param)
        random_forest_regressor = RandomForestRegressor(
            n_jobs=-1, oob_score=True, n_estimators=param['n_estimators'], max_depth=param['max_depth'],
            min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf']
        )
        random_forest_regressor.fit(X_train, y_train)
        oob_score_ += random_forest_regressor.oob_score_
        pred.append(random_forest_regressor.predict(X_test))

    pred = np.array(pred)
    pred = np.mean(pred, axis=0)
    print('oob_score_:', oob_score_ / len(params))
    print('mape_error:', mean_absolute_percentage_error(pred, y_test) * 100)


def grid_search(train_data, train_label):
    param_test_1 = {
        'n_estimators': [140, 160, 190, 230, 370, 440, 470, 510]
    }

    param_test_2 = {
        'max_depth': [38, 40, 42, 43, 44, 45, 54, 60, 62, 78, 88]
    }

    param_test_3 = {
        'min_samples_split': range(2, 100, 5),
    }

    param_test_4 = {
        'min_samples_leaf': range(1, 100, 5),
    }

    param_test_5 = {
        'n_estimators': [370, 440, 470, 510],
        'max_depth': [38, 42, 43, 54, 60],
        'min_samples_split': range(2, 8, 5),
        'min_samples_leaf': range(1, 7, 5),
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(n_jobs=-1, oob_score=True, ),
        param_grid=param_test_5, scoring='neg_mean_absolute_percentage_error', cv=5, verbose=3
    )

    grid_search.fit(train_data, train_label)

    print(f"best_params: {grid_search.best_params_}, best_score: {grid_search.best_score_}")

    grid_search_result = []
    for params, mean_test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
        params['mean_test_score'] = mean_test_score
        grid_search_result.append(params)

    pd.DataFrame(grid_search_result).to_csv('check.csv')


def opt_randoforest_regressor(train_data, train_label, test_data):
    # base random_forest_regressor
    params = [
        {
            'n_estimators': 370,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 440,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 470,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        {
            'n_estimators': 510,
            'max_depth': 38,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        }
    ]

    pred = []
    for param in params:
        print('\n', param)
        random_forest_regressor = RandomForestRegressor(
            n_jobs=-1, oob_score=True, n_estimators=param['n_estimators'], max_depth=param['max_depth'],
            min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf']
        )
        random_forest_regressor.fit(train_data, train_label)
        print('oob_score_: ', random_forest_regressor.oob_score_)
        pred.append(random_forest_regressor.predict(test_data))

    pred = np.array(pred)
    pred = np.mean(pred, axis=0)
    pd.DataFrame({'predicted_price': pred}, index=test_data.index).to_csv('public_submission.csv')


if __name__ == '__main__':
    train_data = pd.read_csv('custom_train_data.csv', index_col='ID')
    train_label = pd.read_csv('custom_train_label.csv', index_col='ID')
    test_data = pd.read_csv('custom_test_data.csv', index_col='ID')

    for n_components in range(47, 1, -1):
        print('n_components:', n_components)
        pca = PCA(n_components=n_components)
        pca.fit(train_data)

        train_data = pca.transform(train_data)
        test_data = pca.transform(test_data)

        randoforest_regressor_mape(train_data, train_label)
        # opt_randoforest_regressor(train_data, train_label, test_data)
