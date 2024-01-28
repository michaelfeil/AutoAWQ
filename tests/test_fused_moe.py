import torch

torch.set_grad_enabled(False)
torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.float16)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

import time
from awq.modules.fused.moe import FusedMixtralSparseMoeBlock
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

config = MixtralConfig()

block = MixtralSparseMoeBlock(config)
fused = FusedMixtralSparseMoeBlock(config)

for i in range(config.num_local_experts):
    fused.ws[i] = torch.cat((
        block.experts[i].w1.weight.data,
        block.experts[i].w3.weight.data,
    ), dim=0)
    fused.w2s[i] = block.experts[i].w2.weight.data

def _run_profile(fn, inputs, rounds=100):
    start_time = time.perf_counter()
    torch.cuda.synchronize()

    for _ in range(rounds):
        states, router_logits = fn(inputs)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return (end_time - start_time) / rounds, states, router_logits

# [batch_size, seq_len, hidden_dim]
inputs = torch.randn((1, 64, config.hidden_size))

block_time, states_block, router_block = _run_profile(block.forward, inputs)
fused_time, states_fused, router_fused = _run_profile(fused.forward, inputs)

print(block_time, fused_time, block_time / fused_time)
print((states_fused - states_block).mean().abs())
print((states_fused - states_block).abs().max())