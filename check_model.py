import torch

ckpt = torch.load('outputs/overfit/best_model.pth', map_location='cpu', weights_only=False)
print('Checkpoint 信息:')
for k, v in ckpt.items():
    if k not in ['model_state_dict', 'optimizer_state_dict']:
        print(f'  {k}: {v}')
print(f'\n模型参数数量: {len(ckpt["model_state_dict"])} 层')
