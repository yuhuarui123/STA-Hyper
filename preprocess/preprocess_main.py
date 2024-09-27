import math
import os
import pickle
import pandas as pd
from datetime import datetime
import os.path as osp

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from generate_poi_feature import save_poi_features
from param_parser import parameter_parser
from utils import get_root_dir
from utils.conf_util import Cfg
from preprocess import (
    ignore_first,
    only_keep_last,
    id_encode,
    remove_unseen_user_poi,
    FileReader,
    generate_hypergraph_from_file
)
import logging


def preprocess_nyc(path: bytes, preprocessed_path: bytes) -> pd.DataFrame:
    raw_path = osp.join(path, 'raw')
    # 读取训练、验证和测试数据集
    df_train = pd.read_csv(osp.join(raw_path, 'NYC_train.csv'))
    df_val = pd.read_csv(osp.join(raw_path, 'NYC_val.csv'))
    df_test = pd.read_csv(osp.join(raw_path, 'NYC_test.csv'))
    # 添加分割标签并合并数据集
    df_train['SplitTag'] = 'train'
    df_val['SplitTag'] = 'validation'
    df_test['SplitTag'] = 'test'
    # 将三个数据集合并成一个大的 DataFrame (df)
    df = pd.concat([df_train, df_val, df_test])
    # 重命名列名
    df.columns = [
        'UserId', 'PoiId', 'PoiCategoryId', 'PoiCategoryCode', 'PoiCategoryName', 'Latitude', 'Longitude',
        'TimezoneOffset', 'UTCTime', 'UTCTimeOffset', 'UTCTimeOffsetWeekday', 'UTCTimeOffsetNormInDayTime',
        'pseudo_session_trajectory_id', 'UTCTimeOffsetNormDayShift', 'UTCTimeOffsetNormRelativeTime', 'SplitTag'
    ]

    # 数据转换
    df['trajectory_id'] = df['pseudo_session_trajectory_id']
    df['UTCTimeOffset'] = df['UTCTimeOffset'].apply(lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"))
    df['UTCTimeOffsetEpoch'] = df['UTCTimeOffset'].apply(lambda x: x.timestamp())
    df['UTCTimeOffsetWeekday'] = df['UTCTimeOffset'].apply(lambda x: x.weekday())
    df['UTCTimeOffsetHour'] = df['UTCTimeOffset'].apply(lambda x: x.hour)
    df['UTCTimeOffsetDay'] = df['UTCTimeOffset'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['UserRank'] = df.groupby('UserId')['UTCTimeOffset'].rank(method='first')

    # be sorted by 'UserId', 'UTCTimeOffset'
    df = df.sort_values(by=['UserId', 'UTCTimeOffset'], ascending=True)

    # id encoding编码
    df['check_ins_id'] = df['UTCTimeOffset'].rank(ascending=True, method='first') - 1
    traj_id_le, padding_traj_id = id_encode(df, df, 'pseudo_session_trajectory_id')

    df_train = df[df['SplitTag'] == 'train']
    poi_id_le, padding_poi_id = id_encode(df_train, df, 'PoiId')
    poi_category_le, padding_poi_category = id_encode(df_train, df, 'PoiCategoryId')
    user_id_le, padding_user_id = id_encode(df_train, df, 'UserId')
    hour_id_le, padding_hour_id = id_encode(df_train, df, 'UTCTimeOffsetHour')
    weekday_id_le, padding_weekday_id = id_encode(df_train, df, 'UTCTimeOffsetWeekday')

    # save mapping logic保存编码映射逻辑
    with open(osp.join(preprocessed_path, 'label_encoding.pkl'), 'wb') as f:
        pickle.dump([
            poi_id_le, poi_category_le, user_id_le, hour_id_le, weekday_id_le,
            padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id
        ], f)

    # ignore the first for train/validate/test and keep the last for validata/test
    df = ignore_first(df)
    df = only_keep_last(df)
    return df


def preprocess_tky_ca(cfg: Cfg, path: bytes) -> pd.DataFrame:
    if cfg.dataset_args.dataset_name == 'tky':
        raw_file = 'dataset_TSMC2014_TKY.txt'
    else:
        raw_file = 'dataset_gowalla_ca_ne.csv'

    FileReader.root_path = path
    data = FileReader.read_dataset(raw_file, cfg.dataset_args.dataset_name)
    data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
    data = FileReader.split_train_test(data)

    # 对于ca数据集，经过一轮筛选后，仍然存在很多低频率的泊位和用户，所以要重复筛选两次
    if cfg.dataset_args.dataset_name == 'ca':
        data = FileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = FileReader.split_train_test(data)

    data = FileReader.generate_id(
        data,
        cfg.dataset_args.session_time_interval,
        cfg.dataset_args.do_label_encode,
        cfg.dataset_args.only_last_metric
    )
    return data


def preprocess(cfg: Cfg):
    root_path = get_root_dir()
    dataset_name = cfg.dataset_args.dataset_name
    data_path = osp.join(root_path, 'data', dataset_name)
    preprocessed_path = osp.join(data_path, 'preprocessed')
    sample_file = osp.join(preprocessed_path, 'sample.csv')
    train_file = osp.join(preprocessed_path, 'train_sample.csv')
    validate_file = osp.join(preprocessed_path, 'validate_sample.csv')
    test_file = osp.join(preprocessed_path, 'test_sample.csv')

    keep_cols = [
        'check_ins_id', 'UTCTimeOffset', 'UTCTimeOffsetEpoch', 'pseudo_session_trajectory_id',
        'query_pseudo_session_trajectory_id', 'UserId', 'Latitude', 'Longitude', 'PoiId', 'PoiCategoryId',
        'PoiCategoryName', 'last_checkin_epoch_time'
    ]

    if not osp.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    #  1. 数据转换; 2. ID 编码; 3. 训练/验证/测试分割; 4. 移除未见用户或POI
    # Step 1. 预处理原始文件并创建样本文件
    if not osp.exists(sample_file):
        if 'nyc' == dataset_name:
            keep_cols += ['trajectory_id']
            preprocessed_data = preprocess_nyc(data_path, preprocessed_path.encode())
        elif 'tky' == dataset_name or 'ca' == dataset_name:
            preprocessed_data = preprocess_tky_ca(cfg, data_path)
        else:
            raise ValueError(f'Wrong dataset name: {dataset_name} ')

        preprocessed_result = remove_unseen_user_poi(preprocessed_data)
        preprocessed_result['sample'].to_csv(sample_file, index=False)
        preprocessed_result['train_sample'][keep_cols].to_csv(train_file, index=False)
        preprocessed_result['validate_sample'][keep_cols].to_csv(validate_file, index=False)
        preprocessed_result['test_sample'][keep_cols].to_csv(test_file, index=False)

    # Step 2. 生成超图相关数据
    if not osp.exists(osp.join(preprocessed_path, 'ci2traj_pyg_data.pt')):
        generate_hypergraph_from_file(sample_file, preprocessed_path, cfg.dataset_args)

    logging.info('[Preprocess] Done preprocessing.')
