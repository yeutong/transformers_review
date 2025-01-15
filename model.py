# %%
import torch
from torch import nn
from einops import einsum
import numpy as np

# %%
class TokEmbeding(nn.Module):
    def __init__(self, d_token, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn((d_token, d_model)))
        self.b_E = nn.Parameter(torch.randn(d_model))

        
    def forward(self, x):
        "shape x: batch seq"
        x = self.W_E[x] # batch seq d_model
        x += self.b_E
        return x

class PosEmbed(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.W_in(x))
        return self.W_out(x)

class SelfAttn(nn.Module):
    def __init__(self, d_model, d_head, n_head):
        super().__init__()
        self.d_head = d_head
        self.eps = 1e-5

        self.W_Q = nn.Parameter(torch.randn((n_head, d_model, d_head)))
        self.b_Q = nn.Parameter(torch.randn((n_head, d_head)))
        self.W_K = nn.Parameter(torch.randn((n_head, d_model, d_head)))
        self.b_K = nn.Parameter(torch.randn((n_head, d_head)))
        self.W_V = nn.Parameter(torch.randn((n_head, d_model, d_head)))
        self.b_V = nn.Parameter(torch.randn((n_head, d_head)))
        self.W_O = nn.Parameter(torch.randn((n_head, d_model, d_head)))
        self.b_O = nn.Parameter(torch.randn((n_head, d_head)))


    def forward(self, x):
        q = einsum(self.W_Q, x, "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head") + self.b_Q
        k = einsum(self.W_K, x, "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head") + self.b_K
        v = einsum(self.W_V, x, "n_head d_model d_head, batch seq d_model -> batch seq n_head d_head") + self.b_V
        
        qk = einsum(q, k, 'batch seq_q n_head d_head, batch seq_k n_head d_head -> batch n_head seq_q seq_k')
        attn_score = qk / np.sqrt(self.d_head) + self.eps # batch n_head seq_q seq_k

        attn_pattern = torch.softmax(attn_score, dim=-1) # batch n_head seq_q seq_k
        attn_out = einsum(attn_pattern, v, "batch n_head seq_q seq_k, batch seq_k n_head d_head -> batch seq_q n_head d_head") + self.b_O
        
        output = einsum(attn_out, self.W_O, 'batch seq n_head d_head, n_head d_model d_head -> batch seq d_model')
        return output


class Unembeding(nn.Module):
    def __init__(self, d_model, d_token):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_token))
        self.b_U = nn.Parameter(torch.randn(d_token))

    def forward(self, x):
        "shape of x: batch seq d_model"
        x = einsum(self.W_U, x, 'd_model d_token, batch seq d_model -> batch seq d_token')
        x += self.b_U
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-5
        self.beta = nn.Parameter(torch.randn(d_model))
        self.gamma = nn.Parameter(torch.rand(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps) 
        return (norm_x - self.beta) / (self.gamma + self.eps)

class Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = LayerNorm(args.d_model)
        self.attn = SelfAttn(args.d_model, args.d_head, args.n_head)
        self.ln2 = LayerNorm(args.d_model)
        self.mlp = MLP(args.d_model, args.d_mlp)

    def forward(self, x):

        x = self.attn(self.ln1(x))
        x = self.mlp(self.ln2(x))
        return x
    
class ModelArgs:
    d_model = 128
    d_head = 32
    n_head = 4
    d_mlp = 512
    d_token = 10000
    n_layer = 3
    ctx_len = 128

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        layers = [Layer(args) for _ in range(args.n_layer)]
        self.layers = nn.Sequential(*layers)

        self.emb = TokEmbeding(args.d_token, args.d_model)

        self.unemb = Unembeding(args.d_model, args.d_token)

    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)
        x = self.unemb(x)
        return x


# %%
model_args = ModelArgs()
model = GPT(model_args)
# %%
x = torch.randint(0, 10000, (3, 10))
x
# %%
output = model(x)
# %%
output.shape
# %%

data = 
# %%
class TrainArgs:
    n_epoch = 5
    lr = 1e-3

def train(model, args):
    
# %%
