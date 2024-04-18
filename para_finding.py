import warnings

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    train_data = pd.read_csv('custom_train_data.csv', index_col='ID')
    train_label = pd.read_csv('custom_train_label.csv', index_col='ID')
    test_data = pd.read_csv('custom_test_data.csv', index_col='ID')

    cv_params = {'n_estimators': range(900, 1000)}
    other_params = {}

    model = ExtraTreesRegressor(**other_params)
    optimized_GS = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_percentage_error',
                                cv=5, verbose=2, n_jobs=-1)
    optimized_GS.fit(train_data, train_label.values.ravel())

    print('參數最佳取值：{0}'.format(optimized_GS.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GS.best_score_))

    pd.DataFrame({
        'mean_test_score': optimized_GS.cv_results_['mean_test_score'],
        'params': optimized_GS.cv_results_['params']
    }).to_csv('cv_results_.csv')
