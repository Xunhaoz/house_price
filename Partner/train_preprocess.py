from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

pd.set_option("display.max_columns", None)


def one_hot_encoder(cols: list, data: pd.DataFrame):
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(data[cols])
    encoded_df = pd.DataFrame(
        encoded_data.toarray(),
        columns=encoder.get_feature_names_out(cols),
        index=data.index
    )
    return encoded_df


def min_max_scalar(cols: list, data: pd.DataFrame):
    sc = MinMaxScaler()
    sc_matrix = sc.fit_transform(data[cols])
    sc_df = pd.DataFrame(data=sc_matrix, columns=cols, index=data.index)
    return sc_df


if __name__ == '__main__':
    one_hot_list = ["主要用途", "主要建材", "建物型態"]
    min_max_list = ['土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '橫坐標',
                    '縱坐標', '主建物面積', '陽台面積', '附屬建物面積']
    drop_list = ['備註', '使用分區', '車位個數', '縣市', '鄉鎮市區', '路名', 'lat', 'lng', '單價']

    data = pd.read_csv("Custom_training_data.csv", index_col='ID')
    data[["單價"]].to_csv('train_label.csv')

    encoded_df = one_hot_encoder(one_hot_list, data)
    sc_df = min_max_scalar(min_max_list, data)

    data = data.drop(columns=(one_hot_list + min_max_list + drop_list))
    clean_df = pd.concat([data, encoded_df, sc_df], axis=1)

    clean_df.to_csv("train_data.csv")
