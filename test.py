import warnings

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from xgboost import XGBRegressor


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


def permutation_prediction(models: list, models_name: list, train_data: pd.DataFrame, train_label: pd.DataFrame,
                           permutation_num: int):
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
    # extra_params = {'n_estimators': 1067, 'min_samples_leaf': 1, 'min_samples_split': 2, 'max_features': 45,
    #                 'n_jobs': -1}

    rf = RandomForestRegressor(n_jobs=-1,verbose=0)
    bagging = BaggingRegressor(n_jobs=-1, verbose=0)
    extra = ExtraTreesRegressor(n_jobs=-1, verbose=0)
    hist = HistGradientBoostingRegressor(verbose=0)
    xgb = XGBRegressor(**xgb_param)
    lgbm = LGBMRegressor()
    cat = CatBoostRegressor(verbose=0)

    models = [
        rf, bagging, extra, hist, xgb, lgbm, cat
    ]

    models_name = [
        'rf', 'bagging', 'extra', 'hist', 'xgb', 'lgbm', 'cat'
    ]

    record_dict = permutation_prediction(models, models_name, train_data, train_label, 2)
    pd.DataFrame(record_dict).to_csv('assemble_learning.csv')
