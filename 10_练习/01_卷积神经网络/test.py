"""
Conv2d 数学计算测试脚本
验证 3×3 卷积的数学计算过程
"""

import torch
import torch.nn as nn

print("=" * 60)
print("Conv2d 3×3 卷积数学计算测试")
print("=" * 60)

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# ==================== 1. 定义输入数据 ====================
print("\n【步骤1】定义输入数据")
print("-" * 60)

# 输入: [batch=1, channel=3, height=4, width=4]
# 使用 4×4 的输入，配合 3×3 卷积核和 padding=1，输出也是 4×4
x = torch.tensor([
    [  # Batch 0
        [[ 1,  2,  3,  4],     # Channel 0 (R)
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12],
         [13, 14, 15, 16]],
        [[17, 18, 19, 20],     # Channel 1 (G)
         [21, 22, 23, 24],
         [25, 26, 27, 28],
         [29, 30, 31, 32]],
        [[33, 34, 35, 36],     # Channel 2 (B)
         [37, 38, 39, 40],
         [41, 42, 43, 44],
         [45, 46, 47, 48]]
    ]
], dtype=torch.float32)

print(f"输入形状: {x.shape}")  # [1, 3, 4, 4]
print(f"\n输入数据:")
print(f"Channel 0 (R):\n{x[0, 0].numpy()}")
print(f"Channel 1 (G):\n{x[0, 1].numpy()}")
print(f"Channel 2 (B):\n{x[0, 2].numpy()}")

# ==================== 2. 定义卷积层 ====================
print("\n【步骤2】定义 3×3 卷积层")
print("-" * 60)

# 定义 3×3 卷积: 输入3通道, 输出2通道, padding=1保持尺寸
conv = nn.Conv2d(
    in_channels=3,      # 输入通道
    out_channels=2,     # 输出通道
    kernel_size=3,      # 3×3卷积核
    padding=1,          # 填充1，保持输出尺寸与输入相同
    bias=False          # 不使用偏置
)

# 为了演示，我们手动设置权重
# 权重形状: [out_channels=2, in_channels=3, kernel_h=3, kernel_w=3]
with torch.no_grad():
    # 输出通道0的权重: 3个输入通道，每个是 3×3
    conv.weight[0] = torch.tensor([
        [[ 0.1,  0.2,  0.3],   # R 通道的 3×3 权重
         [ 0.4,  0.5,  0.6],
         [ 0.7,  0.8,  0.9]],
        [[ 0.1,  0.1,  0.1],   # G 通道的 3×3 权重
         [ 0.1,  0.1,  0.1],
         [ 0.1,  0.1,  0.1]],
        [[-0.1, -0.1, -0.1],   # B 通道的 3×3 权重
         [-0.1, -0.1, -0.1],
         [-0.1, -0.1, -0.1]]
    ])
    # 输出通道1的权重
    conv.weight[1] = torch.tensor([
        [[ 0.0,  0.1,  0.0],   # R 通道
         [ 0.1,  0.5,  0.1],
         [ 0.0,  0.1,  0.0]],
        [[ 0.0,  0.0,  0.0],   # G 通道
         [ 0.0,  0.3,  0.0],
         [ 0.0,  0.0,  0.0]],
        [[ 0.0, -0.1,  0.0],   # B 通道
         [-0.1, -0.2, -0.1],
         [ 0.0, -0.1,  0.0]]
    ])

print(f"卷积核形状: {conv.weight.shape}")
print(f"\n卷积核权重:")
for i in range(2):
    print(f"\n  输出通道 {i}:")
    for j in range(3):
        print(f"    输入通道 {j}:\n{conv.weight[i, j].detach().numpy()}")

# ==================== 3. 执行卷积计算 ====================
print("\n【步骤3】执行卷积计算")
print("-" * 60)

output = conv(x)
print(f"输出形状: {output.shape}")  # [1, 2, 4, 4]

# ==================== 4. 手动验证计算 ====================
print("\n【步骤4】手动验证计算")
print("-" * 60)

# 获取权重
w = conv.weight.data

# 计算输出位置 [0, 0, 1, 1] (Batch=0, OutCh=0, H=1, W=1)
# 这个位置对应输入的 3×3 区域 (不考虑padding)
print("\n计算输出 [0, 0, 1, 1] (输出通道0, 位置(1,1)):")
print("对应输入区域 (H=0~2, W=0~2):")

# 手动计算输出通道0，位置(1,1)
# 输入区域: R[0:3, 0:3], G[0:3, 0:3], B[0:3, 0:3]
manual_calc_0_11 = 0.0
print("\n  R 通道输入:")
print(f"    {x[0, 0, 0:3, 0:3].numpy()}")
print("  R 通道权重:")
print(f"    {w[0, 0].numpy()}")
for i in range(3):
    for j in range(3):
        val = x[0, 0, i, j] * w[0, 0, i, j]
        print(f"    x[0,0,{i},{j}] × w[0,0,{i},{j}] = {x[0, 0, i, j].item():.0f} × {w[0, 0, i, j].item():.1f} = {val.item():.2f}")
        manual_calc_0_11 += val

print("\n  G 通道输入:")
print(f"    {x[0, 1, 0:3, 0:3].numpy()}")
print("  G 通道权重:")
print(f"    {w[0, 1].numpy()}")
for i in range(3):
    for j in range(3):
        val = x[0, 1, i, j] * w[0, 1, i, j]
        print(f"    x[0,1,{i},{j}] × w[0,1,{i},{j}] = {x[0, 1, i, j].item():.0f} × {w[0, 1, i, j].item():.1f} = {val.item():.2f}")
        manual_calc_0_11 += val

print("\n  B 通道输入:")
print(f"    {x[0, 2, 0:3, 0:3].numpy()}")
print("  B 通道权重:")
print(f"    {w[0, 2].numpy()}")
for i in range(3):
    for j in range(3):
        val = x[0, 2, i, j] * w[0, 2, i, j]
        print(f"    x[0,2,{i},{j}] × w[0,2,{i},{j}] = {x[0, 2, i, j].item():.0f} × {w[0, 2, i, j].item():.1f} = {val.item():.2f}")
        manual_calc_0_11 += val

print(f"\n  手动计算结果: {manual_calc_0_11.item():.4f}")
print(f"  PyTorch 结果: {output[0, 0, 1, 1].item():.4f}")
print(f"  验证: {'✓ 通过' if torch.isclose(manual_calc_0_11, output[0, 0, 1, 1]) else '✗ 失败'}")

# ==================== 5. 完整输出结果 ====================
print("\n【步骤5】完整输出结果")
print("-" * 60)

print("\nPyTorch 卷积输出:")
for i in range(2):
    print(f"\n输出通道 {i}:")
    print(output[0, i].detach().numpy())

# ==================== 6. 完整手动计算验证 ====================
print("\n【步骤6】完整手动计算验证所有位置")
print("-" * 60)

# 手动计算所有输出
manual_output = torch.zeros_like(output)
for b in range(1):  # batch
    for oc in range(2):  # 输出通道
        for h in range(4):  # 高度
            for w_pos in range(4):  # 宽度
                # 3×3 卷积: 遍历卷积核的每个位置
                for ic in range(3):  # 输入通道
                    for kh in range(3):  # 卷积核高度
                        for kw in range(3):  # 卷积核宽度
                            # 计算输入位置（考虑padding）
                            ih = h + kh - 1  # padding=1，所以减1
                            iw = w_pos + kw - 1
                            # 检查边界
                            if 0 <= ih < 4 and 0 <= iw < 4:
                                manual_output[b, oc, h, w_pos] += x[b, ic, ih, iw] * w[oc, ic, kh, kw]

print("\n手动计算输出:")
for i in range(2):
    print(f"\n输出通道 {i}:")
    print(manual_output[0, i].numpy())

# 验证是否相等
max_diff = torch.max(torch.abs(output - manual_output)).item()
print(f"\n最大差异: {max_diff:.8f}")
# 使用更合理的阈值 1e-5，这是浮点数比较的常用容差
print(f"验证结果: {'✓ 全部通过' if max_diff < 1e-5 else '✗ 存在差异'}")

# ==================== 7. 添加 BatchNorm 和 SiLU ====================
print("\n【步骤7】添加 BatchNorm2d 和 SiLU 激活")
print("-" * 60)

# 创建完整的 cv1 模块
cv1 = nn.Sequential(
    nn.Conv2d(3, 2, 3, padding=1, bias=False),
    nn.BatchNorm2d(2),
    nn.SiLU()
)

# 复制卷积权重
cv1[0].weight.data = conv.weight.data

# 设置 BatchNorm 参数（为了演示）
with torch.no_grad():
    cv1[1].weight[:] = 1.0   # gamma
    cv1[1].bias[:] = 0.0     # beta
    cv1[1].running_mean[:] = 0.0
    cv1[1].running_var[:] = 1.0

cv1.eval()  # 评估模式，使用 running statistics

output_full = cv1(x)

print(f"\n完整 cv1 输出形状: {output_full.shape}")
print(f"\n输出通道 0 (经过 BatchNorm + SiLU):")
print(output_full[0, 0].detach().numpy())

# 手动验证 BatchNorm + SiLU
print("\n手动验证 (BatchNorm + SiLU):")
# BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
# 这里 mean=0, var=1, gamma=1, beta=0, eps=1e-5
# 所以: y = x / sqrt(1 + 1e-5) ≈ x
# SiLU: y * sigmoid(y)

bn_output = output / torch.sqrt(torch.tensor(1.0 + 1e-5))
silu_output = bn_output * torch.sigmoid(bn_output)

print(f"卷积后: {output[0, 0, 1, 1].item():.4f}")
print(f"BatchNorm后: {bn_output[0, 0, 1, 1].item():.4f}")
print(f"SiLU后: {silu_output[0, 0, 1, 1].item():.4f}")
print(f"PyTorch输出: {output_full[0, 0, 1, 1].item():.4f}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
