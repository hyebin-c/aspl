import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/hbcho/zero/palm')

# HTSATWrapper를 import하기 위해 필요한 설정
from pengi.models.htsat import HTSATWrapper

# HTSATWrapper 인스턴스 생성 (prompt_audio_dim=1)
model = HTSATWrapper(
    sample_rate=16000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=50,
    fmax=8000,
    classes_num=10,
    out_emb=2048,
    specaug=True,
    mixup=False,
    use_precomputed_melspec=False,
    prompt_audio_dim=1  # Enable soft prompts
)

print("=" * 80)
print("HTSATWrapper - All Parameters")
print("=" * 80)
total_params = 0
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
    total_params += param.numel()

print("\n" + "=" * 80)
print("Learnable Soft Prompt Parameters Only")
print("=" * 80)
soft_prompt_params = 0
for name, param in model.named_parameters():
    if 'gamma' in name or 'beta' in name:
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
        soft_prompt_params += param.numel()

print("\n" + "=" * 80)
print(f"Total soft prompt parameters: {soft_prompt_params}")
print("=" * 80)

# Optimizer 생성해서 gamma, beta가 포함되는지 확인
print("\n" + "=" * 80)
print("Optimizer에 포함되는 Parameters")
print("=" * 80)

# 방법 1: HTSATWrapper의 특정 parameters만 optimizer에 포함
gamma_beta_params = []
for name, param in model.named_parameters():
    if 'gamma' in name or 'beta' in name:
        gamma_beta_params.append(param)

optimizer = torch.optim.SGD(gamma_beta_params, lr=0.01, momentum=0.9)

print(f"Optimizer에 포함된 parameter groups: {len(optimizer.param_groups)}")
for group_idx, group in enumerate(optimizer.param_groups):
    print(f"  Group {group_idx}:")
    print(f"    LR: {group['lr']}")
    print(f"    Parameters: {len(group['params'])}")
    for param in group['params']:
        print(f"      Shape: {param.shape}, requires_grad={param.requires_grad}")

print("\n" + "=" * 80)
print("방법 2: model 전체의 parameters를 optimizer에 포함")
print("=" * 80)
optimizer2 = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"Optimizer에 포함된 총 parameters: {sum(len(group['params']) for group in optimizer2.param_groups)}")

print("\n" + "=" * 80)
print("gamma, beta의 초기값 확인")
print("=" * 80)
if hasattr(model, 'gamma'):
    print(f"gamma shape: {model.gamma.shape}")
    print(f"gamma 초기값 (처음 5개): {model.gamma.view(-1)[:5]}")
    print(f"gamma requires_grad: {model.gamma.requires_grad}")

if hasattr(model, 'beta'):
    print(f"beta shape: {model.beta.shape}")
    print(f"beta 초기값 (처음 5개): {model.beta.view(-1)[:5]}")
    print(f"beta requires_grad: {model.beta.requires_grad}")
