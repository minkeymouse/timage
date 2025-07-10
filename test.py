import torch
print(torch.__version__)               # should be 2.7.0+ or nightly
print(torch.cuda.get_arch_list())      # should list 'sm_120'
print(torch.randn(1).cuda())           # should run without the earlier warning

from mamba import mamba_ssm
