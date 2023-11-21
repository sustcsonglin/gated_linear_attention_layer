
import torch

# recurrence
def naive_fwd(v1, v2, g1, g2, q):
    B, L, D = v1.shape
    hidden_state = torch.zeros(B, D, D).to(v1)

    O = torch.empty(B, L, D).to(v1)

    for i in range(L):
        hidden_state = hidden_state * (g1[:, i].unsqueeze(-2) * g2[:, i].unsqueeze(-1)) + v1[:, i].unsqueeze(-2) * v2[:, i].unsqueeze(-1)        
        output = (hidden_state * q[:, i].unsqueeze(-2)).sum(-1)
        O[:, i] = output

    return O


       

def naive_fwd_v2(v1, v2, g1, g2, q):    
    B, L, D = v1.shape

    acc_g = torch.cumprod(g1, dim=-2)

    acc_g2 = torch.cumprod(g2, dim=-2)

    q = q * acc_g 

    v1 = v1 / acc_g  
    
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1)

    attn = (q @ v1.transpose(-1, -2)).masked_fill_(mask, 0) 

    v2 = v2 / acc_g2

    return (attn @ v2) * acc_g2    



if __name__ == "__main__":
    B = 4
    L = 1024
    D = 16
    

    v1 = torch.randn(B, L, D).cuda().exp().requires_grad_(True)
    v2 = torch.randn(B, L, D).cuda().exp().requires_grad_(True)
    g1 = torch.randn(B, L, D).cuda().sigmoid().requires_grad_(True)
    g2 = torch.ones(B, L, D).cuda().requires_grad_(True)

    q = torch.randn(B, L, D).cuda().requires_grad_(True)
    output1 = naive_fwd(v1, v2, g1, g2, q)
    output2 = naive_fwd_v2(v1, v2, g1, g2, q)
    breakpoint()


    

    


    
            

        

        

    
    
    
    
