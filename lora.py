import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_shape = x.shape

        if len(original_shape) <= 2:
            return (self.lora_B @ self.lora_A @ x.T).T * self.scaling
        else:
            x_flat = x.reshape(-1, original_shape[-1])
            output_flat = (self.lora_B @ self.lora_A @ x_flat.T).T * self.scaling
            return output_flat.reshape(*original_shape[:-1], self.out_features)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.0, trainable_orig=False):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank=rank, alpha=alpha)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for param in self.linear.parameters():
            param.requires_grad = trainable_orig

    def forward(self, x):
        return self.linear(x) + self.dropout(self.lora(x))


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.0, trainable_orig=False,
                        target_modules=None, exclude_modules=None):
    """将模型中的线性层替换为LoRA增强的版本"""
    if target_modules is None:
        target_modules = ["Linear"]
    else:
        # 确保大小写一致性
        target_modules = [t.lower() for t in target_modules]

    if exclude_modules is None:
        exclude_modules = []

    replaced_modules = []

    # 遍历所有模块，包括嵌套模块
    for name, module in list(model.named_modules()):
        # 检查是否符合条件且是叶节点模块
        is_target = any(target.lower() in module.__class__.__name__.lower() for target in target_modules)
        is_excluded = any(exclude in name for exclude in exclude_modules)
        is_linear = isinstance(module, nn.Linear)

        if is_target and not is_excluded and is_linear:
            # 找到其父模块
            parent_name, module_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent_module = model

            if parent_name:
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)

            # 替换为LoRA版本
            original_module = getattr(parent_module, module_name)
            setattr(parent_module, module_name, LinearWithLoRA(
                original_module, rank=rank, alpha=alpha, dropout=dropout, trainable_orig=trainable_orig
            ))
            replaced_modules.append(name)

    print(f"已找到 {len(list(model.named_modules()))} 个模块")
    print(f"目标类型: {target_modules}, 排除模式: {exclude_modules}")

    return replaced_modules