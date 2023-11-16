# from .full import flash_gret_full
import torch
from .full_no_fuse import compute_inter_chunk_on2

import torch.nn.functional as F




def naive(query, key, value, gk, gv, BLOCK_N ):
    L = query.shape[-2]
    o = torch.zeros_like(value)
    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

    A = torch.zeros(query.shape[0], query.shape[1], query.shape[2], query.shape[2], device=query.device, dtype=query.dtype)

    for i in range(1, L // BLOCK_N):
        q = query[:, :, (i) * BLOCK_N : (i+1) * BLOCK_N, :]
        gk_q = gk[:, :, (i) * BLOCK_N : (i+1) * BLOCK_N, :]
        gv_q = gv[:, :, (i) * BLOCK_N : (i+1) * BLOCK_N, :]

        output = torch.zeros_like(o[:, :, (i) * BLOCK_N : (i+1) * BLOCK_N, :])
                    
        for j in range(0, i):
            k = key[:, :,  j * BLOCK_N : (j+1) * BLOCK_N, :]
            v = value[:, :, j * BLOCK_N : (j+1) * BLOCK_N, :]
            gk_k = gk[:, :, j * BLOCK_N : (j+1) * BLOCK_N, :]        
            gv_k = gv[:, :, j * BLOCK_N : (j+1) * BLOCK_N, :]

            decay = (gk_q[..., :, None, :] - gk_k[..., None, :, :]).exp()
        
            qk = ((q[..., :, None, :] * k[..., None, :, :]) * decay).sum(-1)   

            A[:, :, (i) * BLOCK_N : (i+1) * BLOCK_N, j * BLOCK_N : (j+1) * BLOCK_N] = qk

            decay_v = (gv_q[..., :, None, :] - gv_k[..., None, :, :]).exp()
        
            output += (qk[..., None] * v[..., None, :, :] * decay_v).sum(-2)
        
        o[:, :, i * BLOCK_N : (i+1) * BLOCK_N, :] = output
            

    return A, o 
            
        
        
        

        

        

        
        
    
    
if __name__ == "__main__":
    B = 2
    H = 2
    L = 2048
    D = 128
    
    q = torch.rand(B, H, L, D, device='cuda')
    k = torch.rand(B, H, L, D, device='cuda')
    v = torch.rand(B, H, L, D, device='cuda')

    gk = torch.rand(B, H, L, D, device='cuda')
    gv = torch.rand(B, H, L, D, device='cuda')

    gk = F.logsigmoid(gk) /16
    gv = F.logsigmoid(gv) / 16
    

    o = naive(q, k, v, gk, gv, 64)
    o2 = compute_inter_chunk_on2(q, k, v, gk, gv, 64)

    breakpoint()




    