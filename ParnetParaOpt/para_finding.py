import warnings

import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_data = pd.read_csv('clean_data_minmax_.csv').drop(columns=['單價'])
    train_label = pd.read_csv('clean_data_minmax_.csv')[['單價']]

    cv_params = {'learning_rate': [i / 100 for i in range(1, 40)]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 1047, 'max_depth': 10, 'min_child_weight': 7, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.01, 'reg_alpha': 0.01, 'reg_lambda': 0.01}

    model = XGBRegressor(**other_params)
    optimized_GS = GridSearchCV(
        estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_percentage_error',
        cv=5, verbose=2, n_jobs=-1
    )

    optimized_GS.fit(train_data, train_label.values.ravel())
    print('參數最佳取值：{0}'.format(optimized_GS.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GS.best_score_))

    pd.DataFrame({
        'mean_test_score': optimized_GS.cv_results_['mean_test_score'],
        'params': optimized_GS.cv_results_['params']
    }).to_csv('cv_results_.csv')
