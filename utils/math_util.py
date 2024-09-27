import torch
from torch import Tensor
from torch_sparse.tensor import SparseTensor
from math import radians, cos, sin, asin, sqrt, exp
import pandas as pd
from bisect import bisect
import time

from math import radians, cos, sin, asin, sqrt
import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def cal_slot_distance(value, slots):
    """
    在插槽中计算具有最近下界和上界的值的距离。
    Calculate a value's distance with nearest lower bound and higher bound in slots.
    :param value: The value to be calculated.
    :param slots: values of slots, needed to be sorted.
    :return: normalized distance with lower bound and higher bound,
        and index of lower bound and higher bound.
    """
    time1 = time.time()
    higher_bound = bisect(slots, value)
    time2 = time.time()
    lower_bound = higher_bound - 1
    if higher_bound == len(slots):
        return 1., 0., lower_bound, lower_bound, time2 - time1
    else:
        lower_value = slots[lower_bound]
        higher_value = slots[higher_bound]
        total_distance = higher_value - lower_value
        return (value - lower_value) / total_distance, (
                higher_value - value) / total_distance, lower_bound, higher_bound, time2 - time1


def cal_slot_distance_batch(batch_value, slots):
    """
    Proceed `cal_slot_distance` on a batch of data.
    :param batch_value: a batch of value, size (batch_size, step)
    :param slots: values of slots, needed to be sorted.
    :return: batch of distances and indexes. All with shape (batch_size, step).
    """
    # Lower bound distance, higher bound distance, lower bound, higher bound.

    ld, hd, l, h = [], [], [], []
    time_cost_list = []
    for step in batch_value:
        ld_one, hd_one, l_one, h_one, time_cost = cal_slot_distance(step, slots)
        ld.append(ld_one)
        hd.append(hd_one)
        l.append(l_one)
        h.append(h_one)
        time_cost_list.append(time_cost)
    print(f"total bisect time: {sum(time_cost_list)}")

    return torch.tensor(ld), torch.tensor(hd), torch.tensor(l), torch.tensor(h)


def construct_slots(min_value, max_value, num_slots, type):
    """
    Construct values of slots given min value and max value.
    :param min_value: minimum value.
    :param max_value: maximum value.
    :param num_slots: number of slots to construct.
    :param type: type of slots to construct, 'linear' or 'exp'.
    :return: values of slots.
    """
    if type == 'exp':
        n = (max_value - min_value) / (exp(num_slots - 1) - 1)
        slots = [n * (exp(x) - 1) + min_value for x in range(num_slots)]
        slots.append(n * (num_slots - 1) + n * 100 + min_value)
        return slots
    elif type == 'linear':
        n = (max_value - min_value) / (num_slots - 1)
        slots = [n * x + min_value for x in range(num_slots-1)]
        slots.append(n*(num_slots-1)*100 + min_value)
        return slots


def delta_t_calculate(x_year: Tensor, adj_t: SparseTensor):
    src_years = x_year[adj_t.storage.col()]
    tar_years = x_year[adj_t.storage.row()]
    delta_ts_pre = tar_years - src_years
    src_tar_mult = src_years * tar_years
    delta_ts = torch.where(src_tar_mult == 0, src_tar_mult, delta_ts_pre)
    return delta_ts


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    def row_wise(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c

    if isinstance(lon1, torch.Tensor):
        if not lon1.numel():
            return None
        lon1 = torch.deg2rad(lon1)
        lat1 = torch.deg2rad(lat1)
        lon2 = torch.deg2rad(lon2)
        lat2 = torch.deg2rad(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
    elif isinstance(lon1, pd.Series):
        if not lon1.shape[0]:
            return None
        lon_lat = pd.concat([lon1, lat1, lon2, lat2], axis=1)
        c = lon_lat.apply(lambda x: row_wise(x[0], x[1], x[2], x[3]), axis=1)
    else:
        if pd.isna(lon1) or pd.isna(lat1) or pd.isna(lon2) or pd.isna(lat2):
            return None
        c = row_wise(lon1, lat1, lon2, lat2)

    r = 6371
    return c * r

# ===================================================================================


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def _init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False


def get_trajectory_embeddings(args, trajectory_data, user_id2idx_dict, all_user_embeddings, all_region_embeddings,
                              all_hotness_embeddings, checkin_embedding_model):
    trajectory_id = trajectory_data[0]
    user_idx = user_id2idx_dict[int(trajectory_id.split("_")[0])]
    user_embedding = all_user_embeddings[user_idx]

    region_idx_list = [data[0] for data in trajectory_data[1]]
    hotness_idx_list = [data[1] for data in trajectory_data[1]]

    region_embedding_list = []
    hotness_embedding_list = []
    for i, region_idx in enumerate(region_idx_list):
        region_embedding_list.append(all_region_embeddings[region_idx])
        hotness_embedding_list.append(all_hotness_embeddings[hotness_idx_list[i]])

    checkin_embedding_list = checkin_embedding_model(
        args, user_embedding, region_embedding_list, hotness_embedding_list
    )

    label_poi_idx = [data for data in trajectory_data[2]]
    return label_poi_idx, checkin_embedding_list
