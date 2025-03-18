import torch

# 指定 checkpoint 目录
# checkpoint_path = f"./checkpoints/best_model_with_bitfit/training_args.bin"
checkpoint_path = f"./checkpoints/best_model_without_bitfit/training_args.bin"

# 读取 training_args.bin
training_args = torch.load(checkpoint_path, weights_only=False)

# 访问所有属性
for key, value in training_args.__dict__.items():
    print(f"{key}: {value}")