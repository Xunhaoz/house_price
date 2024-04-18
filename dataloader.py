from pathlib import Path

import numpy as np
import pandas as pd
import twd97
from tqdm import tqdm


class Dataloader:
    def __init__(self):
        self.train = pd.read_csv('data/training_data.csv', index_col='ID')
        self.test = pd.read_csv('data/public_dataset.csv', index_col='ID')

        # pipline
        self.test = self._encoder_pipline(self.test)
        self.train = self._encoder_pipline(self.train)

        self.train, self.test = self._des2cont(self.train, self.test, '主要用途')
        # self.train, self.test = self._des2cont(self.train, self.test, '主要建材')
        self.train, self.test = self._des2cont(self.train, self.test, '建物型態')

        self.train, self.test = self._des2cont(self.train, self.test, '路名')
        self.train, self.test = self._des2cont(self.train, self.test, '鄉鎮市區')
        self.train, self.test = self._des2cont(self.train, self.test, '縣市')

        self.test = self._external_pipline(self.test)
        self.train = self._external_pipline(self.train)

        self.test = self._scalar_pipline(self.test)
        self.train = self._scalar_pipline(self.train)

        # feature filter and (data, label) split
        self.test = self._feature_filter(self.test, False)
        self.train = self._feature_filter(self.train)

    def _des2cont(self, train_df: pd.DataFrame, test_df: pd.DataFrame, keyword: str):
        roads = train_df[keyword].value_counts().keys()
        road_dict = {road: 0 for road in test_df[keyword].value_counts().keys()}
        for road in roads:
            road_dict[road] = train_df[train_df[keyword] == road]['單價'].mean()
        ordered_road_list = sorted(road_dict.items(), key=lambda x: x[1])
        road_dict = {road[0]: index for index, road in enumerate(ordered_road_list)}
        train_df[keyword] = train_df[keyword].apply(lambda x: road_dict[x])
        test_df[keyword] = test_df[keyword].apply(lambda x: road_dict[x])
        return train_df, test_df

    def _encoder_pipline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # df = self._one_hot_encoding(df, '主要用途')
        df = self._one_hot_encoding(df, '主要建材')
        # df = self._one_hot_encoding(df, '建物型態')
        return df

    def _scalar_pipline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._std_scalar(df, '橫坐標')
        df = self._std_scalar(df, '縱坐標')
        df = self._std_scalar(df, '移轉層次')
        df = self._std_scalar(df, '總樓層數')
        df = self._std_scalar(df, '屋齡')
        df = self._std_scalar(df, '車位個數')
        df = self._std_scalar(df, '主要用途')
        # df = self._std_scalar(df, '主要建材')
        df = self._std_scalar(df, '建物型態')
        df = self._std_scalar(df, '路名')
        df = self._std_scalar(df, '鄉鎮市區')
        df = self._std_scalar(df, '縣市')

        # external
        df = self._std_scalar(df, 'ATM資料')
        df = self._std_scalar(df, '便利商店')
        df = self._std_scalar(df, '公車站點資料')
        df = self._std_scalar(df, '國中基本資料')
        df = self._std_scalar(df, '國小基本資料')
        df = self._std_scalar(df, '大學基本資料')
        df = self._std_scalar(df, '捷運站點資料')
        df = self._std_scalar(df, '火車站點資料')
        df = self._std_scalar(df, '腳踏車站點資料')
        df = self._std_scalar(df, '郵局據點資料')
        df = self._std_scalar(df, '醫療機構基本資料')
        df = self._std_scalar(df, '金融機構基本資料')
        df = self._std_scalar(df, '高中基本資料')

        return df

    def _external_pipline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        lat_lng_list = []
        for e, n in zip(df['橫坐標'].values, df['縱坐標'].values):
            lat, lng = twd97.towgs84(e, n)
            lat_lng_list.append([lat, lng])

        external_datas = Path('data/external_data').glob('*.csv')
        for external_data in external_datas:
            statistics_list = []
            data = pd.read_csv(external_data)
            data_lat, data_lng = data['lat'].values, data['lng'].values
            for lat, lng in tqdm(lat_lng_list, desc=external_data.name):
                dist = self.geo_distance((lat, lng), (data_lat, data_lng))
                statistics_list.append((dist < 1).sum())

            df[external_data.name.split('.')[0]] = statistics_list
        return df

    @staticmethod
    def _feature_filter(df: pd.DataFrame, train=True) -> dict:
        df = df.copy()
        d = {}

        if train:
            d['label'] = df[['單價']]
            df = df.drop(columns=['單價'])

        df = df.drop(columns=['使用分區', '備註'])
        d['data'] = df
        return d

    def _one_hot_encoding(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        df = df.copy()
        value_counts = self.train[col_name].value_counts()
        for value in value_counts.keys():
            df[f'{col_name}_{value}'] = df[col_name].apply(lambda x: 1 if x == value else 0)
        df = df.drop(columns=[col_name])
        return df

    @staticmethod
    def _std_scalar(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        df = df.copy()
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
        return df

    @staticmethod
    def geo_distance(a: tuple[float, float], b: tuple[np.array, np.array]) -> np.array:
        earth_radius = 6371.009

        a_lat, a_lng = a
        b_lat, b_lng = b

        d_lat = np.radians(b_lat - a_lat)
        d_lng = np.radians(b_lng - a_lng)

        h = (np.sin(d_lat / 2) ** 2 + np.cos(np.radians(a_lat)) * np.cos(np.radians(b_lat)) * (
                np.sin(d_lng / 2) ** 2))
        d = 2 * earth_radius * np.arctan2(np.sqrt(h), np.sqrt(np.ones_like(h) - h))

        return d


dataloader = Dataloader()
dataloader.train['data'].to_csv('custom_train_data.csv')
dataloader.train['label'].to_csv('custom_train_label.csv')
dataloader.test['data'].to_csv('custom_test_data.csv')
