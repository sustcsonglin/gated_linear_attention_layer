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
from .compute_o import InterChunk_Compute_O
from .compute_qk import InterChunk_Compute_qk

def compute_inter_chunk_on2(q, k, v, gk, gv, chunk_size):

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

    A = InterChunk_Compute_qk.apply(q, k, gk, 64, 32)
    O = InterChunk_Compute_O.apply(A, v, gv, 64, 32)        

    return O 









