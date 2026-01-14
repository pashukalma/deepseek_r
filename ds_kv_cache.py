import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import math, os, time

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm.auto import tqdm

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


'''
In the Multi-Query Layer Attention the query projection maps to full model
dimension. Key, value projections map to just a single head dimension.
Use the repeat function to duplicate the single key and value for all query.
'''
class MQA(torch.nn.Module):
    def __init__(self, ds_model, num_heads, dropout=0.0):
        super().__init__()
        assert ds_model % num_heads == 0, 'ds model divisible by num_heads'
        self.num_heads = num_heads
        self.head_size = ds_model // num_heads

        ''' the Query projection remains the same as the standard MHA '''
        self.W_query_proj = nn.Linear(ds_model, ds_model)
        ''' the Key and Value projections are now single, shared linear layers,
        projecting down to the dimension of a single head (head_size) '''
        self.W_key_proj = nn.Linear(ds_model, self.head_size)
        self.W_value_proj = nn.Linear(ds_model, self.head_size)
        self.W_out_proj = nn.Linear(ds_model, ds_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(1, 1, 1024, 1024)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.W_query_proj(x).view(batch_size, seq_len,
                              self.num_heads, self.head_size).transpose(1, 2)
        k = self.W_key_proj(x).view(batch_size, seq_len, 1,
                              self.head_size).transpose(1, 2)
        v = self.W_value_proj(x).view(batch_size, seq_len, 1,
                              self.head_size).transpose(1, 2)

        ''' the single Key and Value tensors are repeated or broadcast
        to match the number of query heads '''
        k = k.repeat(1, self.num_heads, 1, 1)
        v = v.repeat(1, self.num_heads, 1, 1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                                self.head_size**.5) #(q @ k.transpose(-2, -1))

        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vector = (attn_weights @ v).transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        output = self.W_out_proj(context_vector)
        return output 

ds_model = 512
num_heads = 8
batch_size = 4
seq_len = 64

mqa_layer = MQA(ds_model, num_heads)
x = torch.randn(batch_size, seq_len, ds_model)
output = mqa_layer(x)
print(x.shape)
print(output.shape)



''' In the Group-Query Attention, the Query projection maps to the full
model dimension. the key and value projections map to num_groups*head_size.
Use repeat interleave to match each key and value group with its corresponding
query heads. '''
class GQA(torch.nn.Module):
    def __init__(self, ds_model, num_heads, num_groups, dropout=0.0,
                 max_seq_len: int=0):
        super().__init__()
        assert ds_model % num_heads == 0, 'ds model divisible by num_heads'
        assert num_heads % num_groups == 0, 'num_heads divisible by num_groups'

        self.ds_model = ds_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_size = ds_model // num_heads

        self.W_query_proj = nn.Linear(ds_model, ds_model)
        ''' Instead of creating a single projection (head_size-number of heads)
         we create num_groups projections '''
        self.W_key_proj = nn.Linear(ds_model, self.head_size)
        self.W_value_proj = nn.Linear(ds_model, self.head_size)
        self.W_out_proj = nn.Linear(ds_model, ds_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask', torch.triu(torch.ones(1, 1, 1024, 1024)))

    def _get_causal_mask(self, seq_len, device):
            if self.causal_mask is not None and self.causal_mask.size[-1] >=seq_len:
                return self.causal_mask[:, :, :seq_len, :seq_len]
            return torch.triu(
                torch.ones(1, 1, seq_len, seq_len), diagonal=1).to(device)

    def _register_mask_buffer(self, max_seq_len):
            if max_seq_len >0:
                mask = torch.triu(torch.ones(
                    1, 1, max_seq_len, max_seq_len), dtype=torch.bool).to(device)
                self.register_buffer('causal_mask', mask, persistence=False)
            else:
                self.causal_mask = None

    def forward(self, x):
            batch_size, seq_len, _ = x.shape()

            q = self.W_query_proj(x).view(
              batch_size, seq_len, self.num_groups, self.head_size).transpose(1, 2)
            ''' the input is projected and reshaped into
            num_groups distinct Key and Value groups '''
            k = self.W_key_proj(x).view(
              batch_size, seq_len, self.num_groups, self.head_size).transpose(1, 2)
            v = self.W_value_proj(x).view(
              batch_size, seq_len, self.num_groups, self.head_size).transpose(1, 2)

            heads_per_group = self.num_heads // self.num_groups
            ''' repeat_interleave broadcasts the K/V groups to query heads,
            each of the num_groups of Keys and Values is shared across head
            per group queries '''
            k = k.repeat_interleave(heads_per_group, dim=2)
            v = v.repeat_interleave(heads_per_group, dim=2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            causal_mask = self._get_causal_mask(seq_len, x.device)

            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context = (attn_weights @ v).transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.ds_model)
            return self.W_out_proj(context)


ds_model = 512
num_heads = 32
num_groups = 4
batch_size = 4
seq_len = 64

''' By changing the num_groups, we can move seamlessly from MQA like behavior
(num_groups=1) to MHA-like behavior (num_groups=head_size, number-of-heads) '''
gqa_layer = GQA(ds_model, num_heads, num_groups)
x = torch.randn(batch_size, seq_len, ds_model)
output = mqa_layer(x)
print(x.shape)
print(output.shape)