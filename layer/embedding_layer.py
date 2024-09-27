import math
import torch
from torch import nn
import os.path as osp

from dataset import predataset
from generate_poi_feature import get_map_dict


class CommonEmbedding(nn.Module):
    def __init__(self, num, embed_dim):
        super(CommonEmbedding, self).__init__()

        self.get_embedding = nn.Embedding(num_embeddings=num, embedding_dim=embed_dim, padding_idx=0)

    def forward(self, args, idx_list):
        idx_list = torch.LongTensor([idx_list]).to(device=args.device)
        embed_list = self.get_embedding(idx_list)
        return embed_list


class CheckInEmbedding(nn.Module):
    def __init__(self):
        super(CheckInEmbedding, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, args, user_embedding, region_embedding_list, hotness_embedding_list):
        checkin_embeddings = torch.randn(0, args.checkin_embed_dim).to(device=args.device)
        for i in range(len(region_embedding_list)):
            poi_embedding = torch.cat((hotness_embedding_list[i], region_embedding_list[i]), 0)
            poi_embedding = self.leaky_relu(poi_embedding)

            checkin_embedding = torch.cat((poi_embedding, user_embedding), 0)
            checkin_embedding = checkin_embedding.reshape(1, args.checkin_embed_dim)
            checkin_embeddings = torch.cat((checkin_embeddings, checkin_embedding), 0)
        return checkin_embeddings


class CheckinEmbedding(nn.Module):
    def __init__(self, embed_size, fusion_type, dataset_args):
        super(CheckinEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type

        self.user_embedding = nn.Embedding(dataset_args.num_user + 1, self.embed_size, padding_idx=dataset_args.padding_user_id)
        self.poi_embedding = nn.Embedding(dataset_args.num_poi + 1, self.embed_size, padding_idx=dataset_args.padding_poi_id)
        self.category_embedding = nn.Embedding(dataset_args.num_category + 1, self.embed_size, padding_idx=dataset_args.padding_poi_category)
        self.dayofweek_embedding = nn.Embedding(8, self.embed_size, padding_idx=dataset_args.padding_weekday_id)
        self.hourofday_embedding = nn.Embedding(25, self.embed_size, padding_idx=dataset_args.padding_hour_id)

        if self.fusion_type == 'concat':
            self.output_embed_size = 5 * self.embed_size
        elif self.fusion_type == 'add':
            self.output_embed_size = embed_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
        embedding_list = [
            self.user_embedding(data[..., 0].long()),
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, embed_size, fusion_type, num_edge_type):
        super(EdgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = nn.Embedding(num_edge_type, self.embed_size)
        self.output_embed_size = self.embed_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embed_size = len(embedding_list) * self.embed_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")
