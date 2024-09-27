import logging
import os
import os.path as osp
import datetime
import torch
import random

from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

from generate_poi_feature import get_map_dict, get_poi_features
from layer.sampler import NeighborSampler
from dataset import PreDataset
from model import My_Model
from preprocess.preprocess_main import preprocess
from utils.sys_util import seed_torch, set_logger
from utils.conf_util import Cfg
from utils.pipeline_util import count_parameters, test_step, save_model

if __name__ == '__main__':
    print(os.path)
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # cuda 设置
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # 对于多次运行，seed被替换为随机值
    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.run_args.save_path = f'tensorboard/{current_time}/{cfg.dataset_args.dataset_name}'
    cfg.run_args.log_path = f'log/{current_time}/{cfg.dataset_args.dataset_name}'

    if not osp.isdir(cfg.run_args.save_path):
        os.makedirs(cfg.run_args.save_path)
    if not osp.isdir(cfg.run_args.log_path):
        os.makedirs(cfg.run_args.log_path)

    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])

    # Preprocess data
    print("1.Preprocess data")
    preprocess(cfg)

    # Initialize dataset
    print("2.Initialize dataset")
    predataset = PreDataset(cfg)

    cfg.dataset_args.spatial_slots = predataset.spatial_slots
    cfg.dataset_args.num_user = predataset.num_user
    cfg.dataset_args.num_poi = predataset.num_poi
    cfg.dataset_args.num_category = predataset.num_category
    cfg.dataset_args.padding_poi_id = predataset.padding_poi_id
    cfg.dataset_args.padding_user_id = predataset.padding_user_id
    cfg.dataset_args.padding_poi_category = predataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = predataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = predataset.padding_weekday_id

    # Loading dicts needed below
    user_id2idx_dict = get_map_dict(osp.join(predataset.data_path, 'user_id2idx_dict.csv'))  # 从文件中加载用户ID到索引的映射字典
    user_list = list(user_id2idx_dict.keys())  # 创建了一个用户 ID 的列表
    poi_id2idx_dict = get_map_dict(osp.join(predataset.data_path, 'poi_id2idx_dict.csv'))  # 将 POI ID 映射到索引
    hotness2idx_dict = get_map_dict(osp.join(predataset.data_path, 'hotness2idx_dict.csv'))  # 加载了一个热度到索引的映射字典
    region2idx_dict = get_map_dict(osp.join(predataset.data_path, 'region2idx_dict.csv'))  # 加载了一个地区到索引的映射字典
    poi_features = get_poi_features(osp.join(predataset.data_path, 'poi_features2idx.csv'))  # 加载 POI 的特征

    # Initialize neighbor sampler(dataloader)
    sampler_train, sampler_validate, sampler_test = None, None, None

    if cfg.run_args.do_train:

        sampler_train = NeighborSampler(
            predataset.x,
            predataset.edge_index,
            predataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=predataset.edge_t,
            edge_delta_t=predataset.edge_delta_t,
            edge_type=predataset.edge_type,
            sizes=sizes,
            sample_idx=predataset.sample_idx_train,
            node_idx=predataset.node_idx_train,
            edge_delta_s=predataset.edge_delta_s,
            max_time=predataset.max_time_train,
            label=predataset.label_train,
            batch_size=cfg.run_args.batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    if cfg.run_args.do_validate:
        sampler_validate = NeighborSampler(
            predataset.x,
            predataset.edge_index,
            predataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=predataset.edge_t,
            edge_delta_t=predataset.edge_delta_t,
            edge_type=predataset.edge_type,
            sizes=sizes,
            sample_idx=predataset.sample_idx_valid,
            node_idx=predataset.node_idx_valid,
            edge_delta_s=predataset.edge_delta_s,
            max_time=predataset.max_time_valid,
            label=predataset.label_valid,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    if cfg.run_args.do_test:
        sampler_test = NeighborSampler(
            predataset.x,
            predataset.edge_index,
            predataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=predataset.edge_t,
            edge_delta_t=predataset.edge_delta_t,
            edge_type=predataset.edge_type,
            sizes=sizes,
            sample_idx=predataset.sample_idx_test,
            node_idx=predataset.node_idx_test,
            edge_delta_s=predataset.edge_delta_s,
            max_time=predataset.max_time_test,
            label=predataset.label_test,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    # %% ====================== Load model ======================
    model = My_Model(cfg)
    model = model.to(device)
    logging.info(f'[Training] Seed: {seed}')
    logging.info('[Training] Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info(f'[Training] Parameter {name}: {param.size()}, require_grad = {param.requires_grad}')
    logging.info(f'[Training] #Parameters: {count_parameters(model)}')

    if cfg.run_args.do_train:
        current_learning_rate = cfg.run_args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        if cfg.run_args.warm_up_steps:
            warm_up_steps = cfg.run_args.warm_up_steps
        else:
            warm_up_steps = cfg.run_args.max_steps // 2

        init_step = 0
        if cfg.run_args.init_checkpoint:
            # 从检查点目录恢复模型
            # 在yml中手动设置
            logging.info(f'[Training] Loading checkpoint %s...' % cfg.run_args.init_checkpoint)
            checkpoint = torch.load(osp.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            cooldown_rate = checkpoint['cooldown_rate']
            sizes = checkpoint['sizes']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info(f'[Training] Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # 设置有效的数据加载器，因为它将在训练期间进行评估
        logging.info(f'[Training] Initial learning rate: {current_learning_rate}')

        # 循环训练
        best_metrics = 0.0
        global_step = 0
        for eph in range(cfg.run_args.epoch):

            training_logs = []
            if global_step >= cfg.run_args.max_steps:
                break

            for data in tqdm(sampler_train):
                model.train()
                split_index = torch.max(data.adjs_t[1].storage.row()).tolist()
                data = data.to(device)
                input_data = {
                    'x': data.x,
                    'edge_index': data.adjs_t,
                    'edge_attr': data.edge_attrs,
                    'split_index': split_index,
                    'delta_ts': data.edge_delta_ts,
                    'delta_ss': data.edge_delta_ss,
                    'edge_type': data.edge_types
                }

                out, loss = model(input_data, label=data.y[:, 0])
                training_logs.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                summary_writer.add_scalar(f'train/loss_step', loss, global_step)

                if cfg.run_args.do_validate and global_step % cfg.run_args.valid_steps == 0:
                    logging.info(f'[Evaluating] Evaluating on Valid Dataset...')

                    logging.info(f'[Evaluating] Epoch {eph}, step {global_step}:')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(model, data=sampler_validate)
                    summary_writer.add_scalar(f'validate/recall@1', 100*recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/recall@5', 100*recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/recall@10', 100*recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/recall@20', 100*recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    metrics = 4 * recall_res[1] + recall_res[20]
                    # metrics = 4 * acc_res[1] + acc_res[20]
                    # save model based on compositional recall metrics
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        logging.info(f'[Training] Save model at step {global_step} epoch {eph}')
                        save_model(model, optimizer, save_variable_list, cfg.run_args, hparam_dict)
                        best_metrics = metrics

                # learning rate schedule
                if global_step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info(f'[Training] Change learning_rate to {current_learning_rate} at step {global_step}')
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * cfg.run_args.cooldown_rate

                if global_step >= cfg.run_args.max_steps:
                    break
                global_step += 1

            epoch_loss = sum([loss for loss in training_logs]) / len(training_logs)
            logging.info(f'[Training] Average train loss at step {global_step} is {epoch_loss}:')
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Start evaluating on test set...')

        checkpoint = torch.load(osp.join(cfg.run_args.save_path, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])

        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(model, sampler_test)
        num_params = count_parameters(model)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/recall@1': recall_res[1],
            'hparam/recall@5': recall_res[5],
            'hparam/recall@10': recall_res[10],
            'hparam/recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
        }
        logging.info(f'[Evaluating] Test evaluation result : {metric_dict}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        summary_writer.close()
