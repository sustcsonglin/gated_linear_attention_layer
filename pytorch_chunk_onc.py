import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra




@torch.jit.script
def compute_inner(query, key, value, decay_key, decay_value, mask):
    original_dtype = query.dtype
    query = (query * decay_key).double()
    key = key.double() / decay_key.double()

    qk = (query @ key.transpose(-1, -2)).masked_fill_(mask, 0)     
    value = value.double() / decay_value.double()
  
    return ((qk @ value) * decay_value).to(original_dtype)





@torch.jit.script
def compute_inter(query,  decay_key,  memory_cache, decay_value):
    return ((query * decay_key) @ memory_cache) * decay_value    


@triton.jit
def _fwd_recurrence(
    S, p1, p2, 
    O,
    NUM_BLOCK, 
    D_MODEL,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    
    
    S = S + offset_bh * NUM_BLOCK * D_MODEL * D_MODEL + offset_d * D_MODEL * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL * D_MODEL + offset_d * D_MODEL * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] +  D_MODEL * D_MODEL    

    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + D_MODEL     
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + D_MODEL  

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)    
    
    S += D_MODEL * D_MODEL    

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL * D_MODEL

    for i in range(NUM_BLOCK-2):
        p_k = tl.load(p1)
        p_v = tl.load(p2)
        S_i = tl.load(S) 
        acc = acc * p_k[:, None] * p_v[None, :] + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        p1 +=  D_MODEL
        p2 += D_MODEL
        S +=  D_MODEL * D_MODEL
        O +=  D_MODEL * D_MODEL        

@triton.jit
def _bwd_recurrence(
    S, p1, p2, 
    DS, Dp,
    NUM_BLOCK, 
    D_MODEL, 
    BLOCK_MODEL: tl.constexpr
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    S = S + offset_bh * NUM_BLOCK * D_MODEL * D_MODEL + offset_d * D_MODEL * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL * D_MODEL

    DS = DS + offset_bh * NUM_BLOCK * D_MODEL * D_MODEL + offset_d * D_MODEL * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 1) * D_MODEL * D_MODEL

    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + D_MODEL + (NUM_BLOCK - 1) * D_MODEL 

    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + D_MODEL +  (NUM_BLOCK - 1) * D_MODEL 
    
    Dp = Dp + offset_bh * NUM_BLOCK * D_MODEL * D_MODEL + offset_d * D_MODEL * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 1) * D_MODEL * D_MODEL

    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)

    for i in range(NUM_BLOCK - 2, -1, -1):
        S_i = tl.load(S)

        DS_i = tl.load(DS)

        dp_i = DS_i * S_i

        p_key = tl.load(p1)
        p_value = tl.load(p2)

        Dacc = Dacc * p_key[:, None] * p_value[None, :] + DS_i 
                
        tl.store(S, Dacc.to(S.dtype.element_ty))
        
        S -= D_MODEL * D_MODEL
        DS -= D_MODEL * D_MODEL
        p1 -= D_MODEL
        p2 -= D_MODEL
        Dp -= D_MODEL * D_MODEL 




class Chunk_memory_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay_key_last, decay_value_last, to_add, use_triton=False):
        decay_key_last = decay_key_last.contiguous()
        decay_value_last = decay_value_last.contiguous()
        to_add = to_add.contiguous()

        ctx.use_triton = use_triton
        
        B, H, N, D, D = to_add.shape 
        assert D % 32 == 0
        
        output = torch.empty_like(to_add)


        if not use_triton:
            memory = torch.zeros(B, H, D, D).to(to_add)          
            for i in range(N-1):
                memory = memory * decay_key_last[:, :, i].unsqueeze(-1) * decay_value_last[:, :, i].unsqueeze(-2) + to_add[:, :, i]
                output[:, :, i+1] = memory.clone()        
        else:
            
            BLOCK_MODEL = 32
            BLOCK_S = 32
            grid = (B*H, D//BLOCK_MODEL, D//BLOCK_MODEL)
            ctx.grid = grid 
            ctx.BLOCK_MODEL = BLOCK_MODEL

            _fwd_recurrence[grid](
                to_add,  
                decay_key_last,
                decay_value_last,
                output,
                D_MODEL=D, NUM_BLOCK=N,  
                BLOCK_MODEL=BLOCK_MODEL
            )
            

        output[:, :, 0] = 0
        ctx.save_for_backward(output, decay_key_last, decay_value_last)        
        
        return output




    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        output, decay_key_last, decay_value_last = ctx.saved_tensors

        B, H, N, D, D = output.shape 


        Dacc1 = torch.zeros(B, H, D, D).to(output)
        num_block = N

        if not ctx.use_triton:
            D_p1 = torch.empty_like(decay_key_last)
            D_p2 = torch.empty_like(decay_value_last)
             
            for i in range(num_block - 2, -1, -1):
                tmp = (output[:, :, i].clone() * DO[:, :, i+1]) 
                dp1 = tmp.sum(-1)
                dp2 = tmp.sum(-2)
                D_p1[:, :, i+1] = dp1 
                D_p2[:, :, i+1] = dp2
                
                p1 = decay_key_last[:, :, i+1]
                p2 = decay_value_last[:, :, i+1]
                Dacc1 = Dacc1 * p1.unsqueeze(-1) * p2.unsqueeze(-2)  + DO[:, :, i+1] 
                output[:, :, i] = Dacc1.clone()

            

            output[:, :, -1] = 0
            D_p1[:, :, 0] = 0
            D_p2[:, :, 0] = 0

            return D_p1, D_p2, output, None
        
        else:
            grid = ctx.grid 
            BLOCK_MODEL = ctx.BLOCK_MODEL 
            D_p = torch.zeros_like(DO)

            _bwd_recurrence[grid](
                output, decay_key_last, decay_value_last,
                DO, D_p, 
                NUM_BLOCK = num_block, 
                D_MODEL = D, 
                BLOCK_MODEL = BLOCK_MODEL
            )
            output[:, :, -1] = 0
            D_p[:, :, 0] = 0
            return D_p.sum(-1), D_p.sum(-2), output, None 


        
            

        
        

    
    


def torch_chunk_parallel_onc(
    key, value,
    g_key, g_value, 
    query, chunk_size=8, use_triton=True, use_cuda=True  
) -> torch.Tensor:
    '''
    query, query: bf16
    '''
    B, H, L, D_h = query.shape    

    assert L % chunk_size == 0        
    num_block = L // chunk_size

    query = rearrange(query, 'b h (n c) d -> b h n c d', c = chunk_size)
    key = rearrange(key, 'b h (n c) d -> b h n c d', c = chunk_size)
    value = rearrange(value, 'b h (n c) d -> b h n c d', c = chunk_size)

    g_key = rearrange(g_key, 'b h (n c) d -> b h n c d', c = chunk_size)
    g_value = rearrange(g_value, 'b h (n c) d -> b h n c d', c = chunk_size)
    
    g_key_cumsum = g_key.cumsum(-2)
    g_value_cumsum = g_value.cumsum(-2)

    reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()
    reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()
    
    to_add = (key * reduce_chunk_key).transpose(-1, -2) @  (value * reduce_chunk_value)

    decay_key = (g_key_cumsum).exp()
    decay_value = (g_value_cumsum).exp()

    decay_key_last = decay_key[..., -1, :]
    decay_value_last = decay_value[..., -1, :]

    memory_cache = Chunk_memory_update.apply(decay_key_last, decay_value_last, to_add,use_triton)    



    inter_chunk_contribution = compute_inter(query,  decay_key,  memory_cache, decay_value)
    
    if not use_cuda:
        mask = torch.triu(torch.ones(chunk_size, chunk_size, device=query.device, dtype=torch.bool), diagonal=1)
        inner_chunk_contribution = compute_inner(query, key, value, decay_key, decay_value, mask) 
    else:
        inner_chunk_contribution, _ = cuda_compute_intra(query, key, value, g_key_cumsum, g_value_cumsum)
    
    output = inter_chunk_contribution + inner_chunk_contribution    
    return rearrange(output, 'b h n c d -> b h (n c) d')



def torch_recurrent_on(v1, v2, g1, g2, q):
    B, H, L, D = v1.shape
    hidden_state = torch.zeros(B, H, D, D).to(v1)

    O = torch.empty(B, H, L, D).to(v1)

    for i in range(L):
        hidden_state = hidden_state * (g1[:, :, i].unsqueeze(-2) + g2[:, :, i].unsqueeze(-1)).exp() + v1[:, :, i].unsqueeze(-2) * v2[:, :, i].unsqueeze(-1)        
        output = (hidden_state * q[:, :, i].unsqueeze(-2)).sum(-1)
        O[:, :, i] = output


    return O

if __name__ == "__main__":
    B = 1
    H = 8
    L = 32
    D = 128
    chunk_size = 16
    device = "cuda"
    requires_grad = True





    v1 = (torch.randn(B, H,  L, D) ).cuda().requires_grad_(requires_grad)  
    v2 = (torch.randn(B, H, L, D) ).cuda().requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  L, D).cuda().uniform_(0.9, 0.99).log().requires_grad_(requires_grad)
    g2 = torch.randn(B, H, L, D).cuda().uniform_(0.9, 0.99).log().requires_grad_(requires_grad)
    # g1 = torch.zeros(B, H, L, D)
    # g2 = torch.zeros(B, H, L, D)
    q = (torch.randn(B, H, L, D) * 5).cuda().requires_grad_(requires_grad)  







    output1 = torch_recurrent_on(v1, v2, g1, g2, q)
    output1.sum().backward()
    target = [v1, v2, g1, g2, q]
    grad1= [ ]
    for v in target:
        grad1.append(v.grad.clone())
        v.grad.zero_()

    

    output2 = torch_chunk_parallel_onc(v1, v2, g1, g2, q, chunk_size=chunk_size,use_triton=True)
    output2.sum().backward()
    grad2= [ ]
    for v in target:
        grad2.append(v.grad.clone())
        v.grad.zero_()


    print( (output1 - output2).abs().max())

    for g1, g2 in zip(grad1, grad2):
        print( (g1 - g2).abs().max())

        






    
    
    
    

        
        
        
        
        
    

    
    
    
    
    

    


    
    
    
    


    

    

    

    
