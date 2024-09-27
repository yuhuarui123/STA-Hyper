import os.path as osp
import pickle
import pandas as pd
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import logging
from utils import get_root_dir
from preprocess import id_encode, ignore_first, only_keep_last


class FileReaderBase:
    root_path = get_root_dir()

    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        raise NotImplementedError


class FileReader(FileReaderBase):
    @classmethod
    def read_dataset(cls, file_name, dataset_name):
        file_path = osp.join(cls.root_path, 'raw', file_name)
        if dataset_name == 'ca':
            df = pd.read_csv(file_path, sep=',')
            df['UTCTimeOffset'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
            df['PoiCategoryName'] = df['PoiCategoryId']
        else:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1', header=None)
            df.columns = [
                'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryName', 'Latitude', 'Longitude', 'TimezoneOffset',
                'UTCTime'
            ]
            df['UTCTime'] = df['UTCTime'].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
            df['UTCTimeOffset'] = df['UTCTime'] + df['TimezoneOffset'].apply(lambda x: timedelta(hours=x/60))

        # 将 'UTCTimeOffset' 转换为 datetime 对象（如果还没有转换的话）
        df['UTCTimeOffset'] = pd.to_datetime(df['UTCTimeOffset'])
        # 将 datetime 对象转换为 Unix 时间戳（以秒为单位）
        df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.timestamp())
        # 提取星期几
        df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
        # 提取小时
        df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
        # 提取年月日
        df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
        # 根据 'UserId' 对 'UTCTimeOffset' 排序并排名
        df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')

        logging.info(
            f'[Preprocess - Load Raw Data] min UTCTimeOffset: {min(df["UTCTimeOffset"])}, '
            f'max UTCTimeOffset: {max(df["UTCTimeOffset"])}, #User: {df["UserId"].nunique()}, '
            f'#POI: {df["PoiId"].nunique()}, #check-in: {df.shape[0]}'
        )
        return df

    @classmethod
    def do_filter(cls, df, poi_min_freq, user_min_freq):

        # 根据 POI 和用户的频率过滤数据,poi_min_freq=9,user_min_freq=9
        poi_count = df.groupby('PoiId')['UserId'].count().reset_index()
        df = df[df['PoiId'].isin(poi_count[poi_count['UserId'] > poi_min_freq]['PoiId'])]
        user_count = df.groupby('UserId')['PoiId'].count().reset_index()
        df = df[df['UserId'].isin(user_count[user_count['PoiId'] > user_min_freq]['UserId'])]

        logging.info(
            f"[Preprocess - Filter Low Frequency User] User count: {len(user_count)}, "
            f"Low frequency user count: {len(user_count[user_count['PoiId'] <= user_min_freq])}, "
            f"ratio: {len(user_count[user_count['PoiId'] <= user_min_freq]) / len(user_count):.5f}"
        )
        logging.info(
            f"[Preprocess - Filter Low Frequency POI] POI count: {len(poi_count)}, "
            f"Low frequency POI count: {len(poi_count[poi_count['UserId'] <= poi_min_freq])}, "
            f"ratio: {len(poi_count[poi_count['UserId'] <= poi_min_freq]) / len(poi_count):.5f}"
        )
        return df

    @classmethod
    def split_train_test(cls, df, is_sorted=False):
        # 将数据集分割为训练集、验证集和测试集
        if not is_sorted:
            df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

        df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')
        df['SplitTag'] = 'train'
        total_len = df.shape[0]
        validation_index = int(total_len * 0.8)
        test_index = int(total_len * 0.9)
        df = df.sort_values(by='UTCTimeOffset', ascending=True)
        df.iloc[validation_index:test_index]['SplitTag'] = 'validation'
        df.iloc[test_index:]['SplitTag'] = 'test'
        df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')

        # 当签入记录与前一次签入和后一次签入的间隔都大于24小时时，过滤掉签入记录
        df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)
        isolated_index = []
        for idx, diff1, diff2, user, user1, user2 in zip(
            df.index,
            df['UTCTimeOffset'].diff(1),
            df['UTCTimeOffset'].diff(-1),
            df['UserId'],
            df['UserId'].shift(1),
            df['UserId'].shift(-1)
        ):
            if pd.isna(diff1) and abs(diff2.total_seconds()) > 86400 and user == user2:
                isolated_index.append(idx)
            elif pd.isna(diff2) and abs(diff1.total_seconds()) > 86400 and user == user1:
                isolated_index.append(idx)
            if abs(diff1.total_seconds()) > 86400 and abs(diff2.total_seconds()) > 86400 and user == user1 and user == user2:
                isolated_index.append(idx)
            elif abs(diff2.total_seconds()) > 86400 and user == user2 and user != user1:
                isolated_index.append(idx)
            elif abs(diff1.total_seconds()) > 86400 and user == user1 and user != user2:
                isolated_index.append(idx)
        df = df[~df.index.isin(set(isolated_index))]

        logging.info('[Preprocess - Train/Validate/Test Split] Done.')
        return df

    @classmethod
    def generate_id(cls, df, session_time_interval, do_label_encode=True, only_last_metric=True):
        # 生成 ID 和标签编码
        df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

        # 生成伪会话轨迹(临时的)
        start_id = 0
        pseudo_session_trajectory_id = [start_id]
        start_user = df['UserId'].tolist()[0]
        time_interval = []
        for user, time_diff in tqdm(zip(df['UserId'], df['UTCTimeOffset'].diff())):
            if pd.isna(time_diff):
                time_interval.append(None)
                continue
            elif start_user != user:
                # difference user
                start_id += 1
                start_user = user
            elif time_diff.total_seconds() / 60 > session_time_interval:
                # same user, beyond interval
                start_id += 1
            time_interval.append(time_diff.total_seconds() / 60)
            pseudo_session_trajectory_id.append(start_id)

        assert len(pseudo_session_trajectory_id) == len(df)

        # 进行标签编码
        if do_label_encode:
            df_train = df[df['SplitTag'] == 'train']
            # todo check if result will be influenced by padding id (nyc use len(), but tky and ca use 0)
            poi_id_le, padding_poi_ie = id_encode(df_train, df, 'PoiId', padding=0)
            poi_category_le, padding_poi_category = id_encode(df_train, df, 'PoiCategoryId', padding=0)
            user_id_le, padding_user_id = id_encode(df_train, df, 'UserId', padding=0)
            hour_id_le, padding_hour_id = id_encode(df_train, df, 'UTCTimeOffsetHour', padding=0)
            weekday_id_le, padding_weekday_id = id_encode(df_train, df, 'UTCTimeOffsetWeekday', padding=0)

            with open(osp.join(cls.root_path, 'preprocessed', 'label_encoding.pkl'), 'wb') as f:
                pickle.dump([
                    poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
                    padding_poi_ie, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
                ], f)

        df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
        df['time_interval'] = time_interval
        df['pseudo_session_trajectory_id'] = pseudo_session_trajectory_id

        # 在创建示例时，忽略每个轨迹的第一次签入
        df = ignore_first(df)

        if only_last_metric:
            df = only_keep_last(df)

        ignore_num = len(df[df["SplitTag"] == "ignore"])
        logging.info(f'[Preprocess] ignore sample num: {ignore_num}, ratio: {ignore_num/df.shape[0]}.')

        trajectory_id_count = df.groupby(['pseudo_session_trajectory_id'])['check_ins_id'].count().reset_index()
        check_ins_count = trajectory_id_count[trajectory_id_count['check_ins_id'] == 1]

        logging.info(
            f"[Preprocess] pseudo session trajectory of single check-ins count: {len(check_ins_count)}, "
            f"ratio: {len(check_ins_count) / len(trajectory_id_count)}."
        )
        return df
