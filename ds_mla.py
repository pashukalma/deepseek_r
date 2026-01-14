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


'''  Apply Rotary Positional Encoding, not part of embedding, and
applied to Query and Key vectors '''
class RoPE(nn.Module):
    def __init__(self, head_size, max_seq_len=2048):
      super().__init__()
      theta = 1.0 / (10000 ** torch.arange(0, head_size, 2).float()/head_size)
      self.register_buffer('theta', theta)

      positions = torch.arange(max_seq_len).float().unsqueeze(1)
      frequencies = positions * self.theta.unsqueeze(0)
      self.register_buffer(
              'frequencies_complex',
              torch.polar(torch.ones_like(frequencies), frequencies))

    def forward(self, x):
        seq_len = x.shape[2]
        x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_complex)
        frequencies_complex = self.frequencies_complex[:seq_len, :].\
                                                    unsqueeze(0).unsqueeze(0)
        x_rotated = x_complex * frequencies_complex
        x_rotated = torch.view_as_real(x_rotated)
        x_rotated = x_rotated.flatten(3)
        return x_rotated.type_as(x)

''' The full sota attention mechanism from DeepSeek,
Multi-Head Attention (MLA) with Rotational Positional Encoding '''
class MLAAttention(nn.Module):
      def __init__(self, ds_model, num_heads, d_latent, d_rope, dropout=0.0,
                   max_seq_len=2048):
        super().__init__()
        assert ds_model % num_heads == 0, 'ds model divisible by num_heads'
        self.ds_model = ds_model
        self.num_heads = num_heads
        self.head_size = ds_model // num_heads
        self.d_latent = d_latent
        self.d_rope = d_rope

        self.W_query_content = nn.Linear(ds_model, ds_model)
        self.W_dkv_content = nn.Linear(ds_model, d_latent)
        self.W_uk_content = nn.Linear(d_latent, ds_model)
        self.W_uv_content = nn.Linear(d_latent, ds_model)

        self.W_k_pos = nn.Linear(ds_model, d_rope * num_heads)
        self.W_q_pos = nn.Linear(ds_model, d_rope * num_heads)

        self.rope = RoPE(d_rope, max_seq_len)

        self.W_out_proj = nn.Linear(ds_model, ds_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(
            torch.ones(1, 1, max_seq_len, max_seq_len), diagonal=1).bool())

      def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q_c = self.W_query_content(x).view(
            batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        c_kv = self.W_dkv_content(x)
        k_c = self.W_uk_content(c_kv).view(
            batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        v_c = self.W_uv_content(c_kv).view(
            batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)

        q_r_unrotated = self.W_q_pos(x).view(
            batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        k_r_unrotated = self.W_k_pos(x).view(
            batch_size, seq_len, self.num_heads, self.head_size).transpose(1,2)
        q_r = self.rope(q_r_unrotated)
        k_r = self.rope(k_r_unrotated)

        content_scores = torch.matmul(
            q_c, k_c.transpose(-2, -1)) / (self.head_size **0.5)
        position_scores = torch.matmul(
            q_r, k_r.transpose(-2, -1)) / (self.d_rope **0.5)
        attn_scores = content_scores + position_scores
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len], float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ v_c).transpose(1, 2).contiguous().\
                            view(batch_size, seq_len, self.ds_model)
        output = self.W_out_proj(context_vector)
        return output