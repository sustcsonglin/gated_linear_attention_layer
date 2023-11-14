import torch 
import time
import math
from typing import Tuple, Union, Optional


import torch
import torch.nn.functional as F
from einops import rearrange

import torch

import triton
import triton.language as tl

import numpy as np
import math
from .fn_only_gk import FlashGRet
from .fn_only_gv import FlashGRet_O

def intra_chunk_computation(q, k, v, gk, gv):
    A = FlashGRet.apply(q, k, gk)
    chunk_size = v.shape[-2]
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    A.masked_fill_(mask, 0)
    return FlashGRet_O.apply(A, v, gv)

