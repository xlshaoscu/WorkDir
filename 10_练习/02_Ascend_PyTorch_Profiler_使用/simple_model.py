"""
简单的神经网络模型 - 用于线性回归任务
支持NPU/GPU/CPU自动切换和模型多格式保存
集成 Ascend PyTorch Profiler 性能分析
"""

import os
import logging
import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PretrainedConfig

# NPU自动迁移导入（仅需导入即可自动迁移）
try:
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

# Ascend PyTorch Profiler 导入
try:
    from torch_npu.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    profile = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================
class SimpleConfig(PretrainedConfig):
    """模型配置类"""
    model_type = "simple"
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 50,
        output_dim: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


# =============================================================================
# 模型类
# =============================================================================
class SimpleModel(PreTrainedModel):
    """简单的神经网络模型"""
    config_class = SimpleConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config: SimpleConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =============================================================================
# 数据集类
# =============================================================================
class SimpleDataset(Dataset):
    """生成简单的线性数据: y = 2x1 + 3x2 + 噪声"""
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(2)
        y = 2 * x[0] + 3 * x[1] + torch.randn(1) * 0.1
        return x, y


# =============================================================================
# 工具函数
# =============================================================================
def get_device() -> str:
    """检测并返回可用的计算设备"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        logger.info("检测到NPU设备，使用NPU运行")
        return "npu"
    elif torch.cuda.is_available():
        logger.info("检测到CUDA设备，使用GPU运行")
        return "cuda"
    else:
        logger.info("未检测到NPU/GPU，使用CPU运行")
        return "cpu"


def create_dataloader(
    num_samples: int = 10000,
    batch_size: int = 64
) -> DataLoader:
    """创建数据加载器"""
    dataset = SimpleDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    prof=None
) -> float:
    """训练一个epoch，返回平均损失，支持Profiler step"""
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每个step后调用profiler step
        if prof is not None:
            prof.step()
    
    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 50,
    device: str = "cpu",
    log_interval: int = 10,
    profiler_output_dir: str = "./profiler_output"
) -> None:
    """训练模型，自动启用Profiler性能分析"""
    logger.info("开始训练...")
    
    # 配置Profiler（默认启用）
    prof = None
    if PROFILER_AVAILABLE:
        os.makedirs(profiler_output_dir, exist_ok=True)
        logger.info(f"启用 Ascend PyTorch Profiler，输出目录: {profiler_output_dir}")
        
        activities = [ProfilerActivity.CPU]
        if device == "npu":
            activities.append(ProfilerActivity.NPU)
        elif device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        
        # 计算总step数，设置合理的采集周期
        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * epochs
        
        # 设置采集策略：跳过前5个step，采集接下来的10个step
        wait_steps = min(5, total_steps // 10)
        active_steps = min(10, total_steps // 5)
        
        prof = profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait_steps, warmup=2, active=active_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
        prof.start()
    else:
        logger.warning("Ascend PyTorch Profiler 不可用，跳过性能分析")
    
    try:
        for epoch in range(epochs):
            avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, prof)
            
            if (epoch + 1) % log_interval == 0:
                logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    finally:
        if prof is not None:
            prof.stop()
            logger.info(f"Profiler 数据已保存到: {profiler_output_dir}")


@torch.no_grad()
def inference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """模型推理"""
    model.eval()
    return model(x)


def test_model(
    model: nn.Module,
    device: str,
    test_cases: Optional[torch.Tensor] = None
) -> None:
    """测试模型推理"""
    if test_cases is None:
        test_cases = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    test_cases = test_cases.to(device)
    predictions = inference(model, test_cases)
    
    logger.info("\n推理测试:")
    logger.info(f"输入: {test_cases}")
    logger.info(f"预测: {predictions}")
    logger.info(f"真实值近似: [8.0, 18.0]")


# =============================================================================
# 主程序
# =============================================================================
def main():
    """主函数"""
    # 获取设备
    device = get_device()
    
    # 创建配置和模型
    config = SimpleConfig()
    model = SimpleModel(config).to(device)
    
    # 创建数据加载器
    dataloader = create_dataloader()
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练模型（自动启用Profiler）
    train_model(model, dataloader, optimizer, criterion, device=device)
    
    # 测试模型
    test_model(model, device)


if __name__ == "__main__":
    main()
