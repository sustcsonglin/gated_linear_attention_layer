# from .full import flash_gret_full
import torch



def naive(query, key, value, gk, gv, BLOCK_N ):
    L = query.shape[-2]
    o = torch.zeros_like(value)

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)

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

            decay_v = (gv_q[..., :, None, :] - gv_k[..., None, :, :]).exp()
        
            output += (qk[..., None] * v[..., None, :, :] * decay_v).sum(-2)
        
        o[:, :, i * BLOCK_N : (i+1) * BLOCK_N, :] = output
            

    return o 
            
        
        
        

        

        

        
        
    
    
# if __name__ == "__main__":
#     B = 2
#     H = 2
#     L = 2048
#     D = 128
    
#     q = torch.rand(B, H, L, D, device='cuda')
#     k = torch.rand(B, H, L, D, device='cuda')
#     v = torch.rand(B, H, L, D, device='cuda')

#     gk = torch.rand(B, H, L, D, device='cuda')
#     gv = torch.rand(B, H, L, D, device='cuda')

#     gk = F.logsigmoid(gk).cumsum(-2) // 16
#     gv = F.logsigmoid(gv).cumsum(-2) // 16
    
#     o = flash_gret_full(q, k, v, gk, gv, BLOCK_N = 64)

#     breakpoint()


    