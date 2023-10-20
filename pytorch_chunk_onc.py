import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra
import torch.nn.functional as F



@torch.jit.script
def compute_inner(query, key, value, decay_key, decay_value, mask):
    original_dtype = query.dtype
    decay_key = decay_key.float().exp()
    decay_value = decay_value.float().exp()    
    query = query.float()
    key = key.float()
    value = value.float()

    query = (query * decay_key)
    key = key / decay_key.clamp_min(1e-6)
    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value / decay_value.clamp_min(1e-6)    

    return ((qk @ value) * decay_value).to(original_dtype)




@torch.jit.script
def compute_inter(query,  decay_key,  memory_cache, decay_value):
    return ((query * decay_key.exp()) @ memory_cache) * decay_value.exp()    


@triton.jit
def _fwd_recurrence(
    S, p1, p2, 
    O,
    NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    
    
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] +  D_MODEL_K * D_MODEL_V    

    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + D_MODEL_K     

    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + D_MODEL_V  

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)

    acc += tl.load(S)    
    
    S += D_MODEL_K * D_MODEL_V    

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V

    for i in range(NUM_BLOCK-2):
        p_k = tl.load(p1)
        p_v = tl.load(p2)
        S_i = tl.load(S) 
        acc = acc * p_k[:, None] * p_v[None, :] + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        p1 +=  D_MODEL_K
        p2 += D_MODEL_V
        S +=  D_MODEL_K * D_MODEL_V
        O +=  D_MODEL_K * D_MODEL_V       


## NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL
@triton.jit
def _bwd_recurrence(
    S, p1, p2, 
    DS, Dp1, Dp2, 

    NUM_BLOCK, NUM_SPLIT_K, NUM_SPLIT_V,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
    
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk  
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    # skip the last chunk because it is never used  
    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K 

    # skip the last chunk because it is never used 
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V 

    # skip the last chunk because it is never used  
    # NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V: stride_bh
    # offset_s * D_MODEL_K: find the right split in the K dimension
    Dp1 = Dp1 + offset_bh * NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V + offset_s * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K * NUM_SPLIT_V


    # skip the last chunk because it is never used 
    Dp2 = Dp2 + offset_bh * NUM_BLOCK * D_MODEL_V * NUM_SPLIT_K + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V  * NUM_SPLIT_K

    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32) 

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):

        p_key = tl.load(p1)
        p_value = tl.load(p2)
        S_i = tl.load(S)
        DS_i = tl.load(DS)

        Dacc += DS_i         
        
        dp_i = Dacc * S_i
        
        dp_key = tl.sum(dp_i * p_value[None, :], axis=1)
        tl.store(Dp1, dp_key.to(Dp1.dtype.element_ty))
        dp_value = tl.sum(dp_i * p_key[:, None], axis=0) 
        tl.store(Dp2, dp_value.to(Dp2.dtype.element_ty))

        tl.store(S, Dacc.to(S.dtype.element_ty))        

        Dacc *= p_key[:, None] * p_value[None, :]

        S -= D_MODEL_K * D_MODEL_V 
        DS -= D_MODEL_K * D_MODEL_V 
        p1 -= D_MODEL_K 
        p2 -= D_MODEL_V 
        Dp1 -= D_MODEL_K * NUM_SPLIT_V
        Dp2 -= D_MODEL_V * NUM_SPLIT_K
    



def naive_chunk_memory_update(decay_key_last, decay_value_last, to_add):
    B, H, N, D_k, D_v = to_add.shape 

    output = torch.empty_like(to_add)        
    output[:, :, 0] = 0
    output[:, :, 1] = to_add[:, :, 0]

    for i in range(2, N):
        output[:, :, i] = to_add[:, :, i-1] + decay_key_last[:, :, i-1].unsqueeze(-1) * decay_value_last[:, :, i-1].unsqueeze(-2) * output[:, :, i-1].clone()

    return output


class Chunk_memory_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay_key_last, decay_value_last, to_add, use_triton=False):
        decay_key_last = decay_key_last.contiguous()
        decay_value_last = decay_value_last.contiguous()
        to_add = to_add.contiguous()

        ctx.use_triton = use_triton
        
        B, H, N, D_k, D_v = to_add.shape 

        output = torch.empty_like(to_add)        

        BLOCK_MODEL = 32
    
        assert D_k % 32 == 0
        assert D_v % 32 == 0
        assert D_k == decay_key_last.shape[-1]
        assert D_v == decay_value_last.shape[-1]

        grid = (B*H, D_k//BLOCK_MODEL, D_v//BLOCK_MODEL)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            to_add,  
            decay_key_last,
            decay_value_last,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N,  
            BLOCK_MODEL=BLOCK_MODEL
        )
        
        output[:, :, 0] = 0
        ctx.save_for_backward(output, decay_key_last, decay_value_last)        
        
        return output

    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        output, decay_key_last, decay_value_last = ctx.saved_tensors 

        B, H, N, D_k, D_v = output.shape 

        num_block = N

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 

        # I don't want atomic_add to be used in the backward pass
        # so I add another dimension to the output tensor (D_k/v // BLOCK_MODEL)
        # afterward, I sum over this dimension to get the correct gradient 
        D_p1 = torch.empty(B, H, N, D_v // BLOCK_MODEL, D_k, device=DO.device)
        D_p2 = torch.empty(B, H, N, D_k // BLOCK_MODEL, D_v, device=DO.device)

        _bwd_recurrence[grid](
            output, decay_key_last, decay_value_last,
            DO, D_p1, D_p2, 
            NUM_BLOCK = num_block, NUM_SPLIT_K = D_k // BLOCK_MODEL, NUM_SPLIT_V = D_v // BLOCK_MODEL, 
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL
        )

        output[:, :, -1] = 0
        D_p1[:, :, 0] = 0
        D_p1[:, :, -1] = 0
        D_p2[:, :, 0] = 0
        D_p2[:, :, -1] = 0
        

        return D_p1.sum(-2), D_p2.sum(-2), output, None         
        

@torch.jit.script
def prepare_to_add(g_key, g_value, key, value):
    g_key = F.logsigmoid(g_key)
    g_key_cumsum = g_key.cumsum(-2)
    reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()

    # g_key_cumsum_exp = g_key_cumsum.exp() 

    g_value = F.logsigmoid(g_value)
    g_value_cumsum = g_value.cumsum(-2)
    reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()

    # g_value_cumsum_exp = g_value_cumsum.exp()
    
    to_add = (key * reduce_chunk_key).transpose(-1, -2) @  (value * reduce_chunk_value)

    return to_add, g_key_cumsum, g_value_cumsum
        
    
def torch_chunk_parallel_onc(
    key, value,
    g_key, g_value, 
    query, chunk_size=16, use_triton=True, use_cuda=True  
) -> torch.Tensor:
    '''
    query, query: bf16
    '''
    B, H, L, D_h = query.shape    
    assert chunk_size == 16

    assert L % chunk_size == 0        
    num_block = L // chunk_size

    query = rearrange(query, 'b h (n c) d -> b h n c d', c = chunk_size)
    key = rearrange(key, 'b h (n c) d -> b h n c d', c = chunk_size)
    value = rearrange(value, 'b h (n c) d -> b h n c d', c = chunk_size)

    g_key = rearrange(g_key, 'b h (n c) d -> b h n c d', c = chunk_size)
    g_value = rearrange(g_value, 'b h (n c) d -> b h n c d', c = chunk_size)
    
    # g_key_cumsum = g_key.cumsum(-2)
    # g_value_cumsum = g_value.cumsum(-2)

    # reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()
    # reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()
    
    # to_add = (key * reduce_chunk_key).transpose(-1, -2) @  (value * reduce_chunk_value)

    # decay_key = (g_key_cumsum).exp()
    # decay_value = (g_value_cumsum).exp()

    # decay_key_last = decay_key[..., -1, :]
    # decay_value_last = decay_value[..., -1, :]

    to_add, g_key_cumsum, g_value_cumsum = prepare_to_add(g_key, g_value, key, value)

    decay_key_last = g_key_cumsum[..., -1, :].exp()
    decay_value_last = g_value_cumsum[..., -1, :].exp()

    memory_cache = Chunk_memory_update.apply(decay_key_last, decay_value_last, to_add, use_triton)    
    
    inter_chunk_contribution = compute_inter(query, g_key_cumsum,  memory_cache, g_value_cumsum)
    

    if use_cuda:
        inner_chunk_contribution, _ = cuda_compute_intra(query, key, value, g_key_cumsum, g_value_cumsum)
    else:
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=query.device, dtype=torch.bool), diagonal=1)
        inner_chunk_contribution = compute_inner(query, key, value, g_key_cumsum, g_value_cumsum, mask=mask)
    
    output = inter_chunk_contribution + inner_chunk_contribution    

    return rearrange(output, 'b h n c d -> b h (n c) d')

    # return output


 

def torch_recurrent_on(v1, v2, g1, g2, q):
    B, H, L, D = v1.shape
    hidden_state = torch.zeros(B, H, D, D).to(v1)

    O = torch.empty(B, H, L, D).to(v1)
    g1 = F.logsigmoid(g1)
    g2 = F.logsigmoid(g2)

    for i in range(L):
        hidden_state = hidden_state * (g1[:, :, i].unsqueeze(-2) + g2[:, :, i].unsqueeze(-1)).exp() + v1[:, :, i].unsqueeze(-2) * v2[:, :, i].unsqueeze(-1)        
        output = (hidden_state * q[:, :, i].unsqueeze(-2)).sum(-1)
        O[:, :, i] = output


    return O



if __name__ == "__main__":
    B = 1
    H = 8
    L = 64
    D = 128

    chunk_size = 16
    device = "cuda"
    requires_grad = True

    # verify the graident of chunk memory update 
    # torch.manual_seed(20)
    decay_key_last = torch.randn(B, H, L, D).cuda().sigmoid().requires_grad_(requires_grad)
    decay_value_last = torch.randn(B, H, L, D).cuda().sigmoid().requires_grad_(requires_grad)
    to_add = torch.randn(B, H, L, D, D).cuda().requires_grad_(requires_grad)

    output1 = naive_chunk_memory_update(decay_key_last, decay_value_last, to_add)
    output1.sum().backward(retain_graph=True)
    target = [decay_key_last, decay_value_last, to_add]
    grad1= [ ]
    for v in target:
        grad1.append(v.grad.clone())
        v.grad.zero_()
    

    output2 = Chunk_memory_update.apply(decay_key_last, decay_value_last, to_add, True)
    print( (output1 - output2).abs().max())

    output2.sum().backward()
    grad2= [ ]

    for v in target:
        grad2.append(v.grad.clone())
        v.grad.zero_()
    
    for g1, g2 in zip(grad1, grad2):
        print( (g1 - g2).abs().max())

    breakpoint()



    # v1 = (torch.randn(B, H,  L, D) ).cuda().requires_grad_(requires_grad)  
    # v2 = (torch.randn(B, H, L, D) ).cuda().requires_grad_(requires_grad) 
    # g1 = torch.randn(B, H,  L, D).cuda().uniform_(0, 0.99).log().requires_grad_(requires_grad)
    # g2 = torch.randn(B, H, L, D).cuda().uniform_(0, 0.99).log().requires_grad_(requires_grad)
    # # g1 = torch.zeros(B, H, L, D)
    # # g2 = torch.zeros(B, H, L, D)
    # q = (torch.randn(B, H, L, D) * 5).cuda().requires_grad_(requires_grad)  











    # output1 = torch_recurrent_on(v1, v2, g1, g2, q)
    # output1.sum().backward()
    # target = [v1, v2, g1, g2, q]
    # grad1= [ ]
    # for v in target:
    #     grad1.append(v.grad.clone())
    #     v.grad.zero_()

    

    # output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True)
    # output2.sum().backward()
    # grad2= [ ]
    # for v in target:
    #     grad2.append(v.grad.clone())
    #     v.grad.zero_()



    # print( (output1 - output2).abs().max())

    # for g1, g2 in zip(grad1, grad2):
    #     print( (g1 - g2).abs().max())

        






    
    
    
    

        
        
        
        
        
    

    
    
    
    
    

    


    
    
    
    


    

    

    

    
