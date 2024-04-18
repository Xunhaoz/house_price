import warnings
from tqdm import tqdm
import pandas as pd

import xgboost as xgb
import lightgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score


def gen_permutation(n):
    permutation_res = []
    for permutation in range(1, 2 ** n):
        res_list = [i for i in range(0, n) if (permutation & (2 ** i))]
        permutation_res.append(res_list)
    return permutation_res


def votion_regressor(model, models_name, train_data, train_label):
    e_reg = VotingRegressor([(k, v) for k, v in zip(models_name, model)], n_jobs=-1)
    scores = cross_val_score(e_reg, train_data, train_label.values.ravel(),
                             scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
    return -scores.mean() * 100


def permutation_prediction(models: list, models_name: list, train_data: pd.DataFrame, train_label: pd.DataFrame):
    record_dict = {
        'permutation': [],
        'score': []
    }

    for permutation in tqdm(gen_permutation(6)):
        selected_model = [models[i] for i in permutation]
        selected_models_name = [models_name[i] for i in permutation]
        score = votion_regressor(selected_model, selected_models_name, train_data, train_label)

        record_dict['permutation'].append(selected_models_name)
        record_dict['score'].append(score)

    return record_dict


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    train_data = pd.read_csv('custom_train_data.csv', index_col='ID')
    train_label = pd.read_csv('custom_train_label.csv', index_col='ID')
    test_data = pd.read_csv('custom_test_data.csv', index_col='ID')

    xgb_param = {'learning_rate': 0.03, 'n_estimators': 792, 'max_depth': 10, 'min_child_weight': 2, 'seed': 0,
                    'subsample': 0.91, 'colsample_bytree': 0.75, 'gamma': 0, 'reg_alpha': 0.2, 'reg_lambda': 0.28}

    random_forest_regressor = RandomForestRegressor(n_jobs=-1)
    bagging_regressor = BaggingRegressor(n_jobs=-1)
    extra_trees_regressor = ExtraTreesRegressor(n_jobs=-1)
    hist_gradient_boosting_regressor = HistGradientBoostingRegressor()
    XGB_regressor = xgb.XGBRegressor(**xgb_param)
    lightgbm = lightgbm.LGBMRegressor(verbose=False)

    models = [
        random_forest_regressor, bagging_regressor, extra_trees_regressor,
        hist_gradient_boosting_regressor, XGB_regressor, lightgbm
    ]

    models_name = [
        'random_forest_regressor', 'bagging_regressor', 'extra_trees_regressor',
        'hist_gradient_boosting_regressor', 'XGB_regressor', 'lightgbm'
    ]

    record_dict = permutation_prediction(models, models_name, train_data, train_label)
    pd.DataFrame(record_dict).to_csv('assemble_learning.csv')
