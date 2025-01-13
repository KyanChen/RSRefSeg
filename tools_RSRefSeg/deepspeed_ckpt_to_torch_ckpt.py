import torch

ckpt_file = 'work_dirs/RSRefSeg-b/epoch_85.pth/mp_rank_00_model_states.pt'

ckpt = torch.load(ckpt_file, map_location='cpu')
ckpt = ckpt['module']
torch.save(ckpt, ckpt_file.replace('mp_rank_00_model_states.pt', 'mp_rank_00_model_states.pth'))