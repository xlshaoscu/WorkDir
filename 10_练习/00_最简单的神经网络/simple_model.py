import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PretrainedConfig
import os
import logging
import inspect

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 自定义配置类
class SimpleConfig(PretrainedConfig):
    model_type = "simple"
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

# 自定义模型类
class SimpleModel(PreTrainedModel):
    config_class = SimpleConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 简单数据集类
class SimpleDataset(Dataset):
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # 生成简单的线性数据: y = 2x1 + 3x2
        x = torch.randn(2)
        y = 2 * x[0] + 3 * x[1] + torch.randn(1) * 0.1
        return x, y

# 训练函数
def train_model(model, dataloader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 推理函数
def inference(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)

if __name__ == "__main__":
    # 检测设备（仅检测，不使用NPU）
    if torch.cuda.is_available():
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 检测到CUDA设备")
    if hasattr(torch, 'npu') and torch.npu.is_available():
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 检测到NPU设备（但不使用）")
    
    # 强制使用CPU运行
    device = "cpu"
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 使用CPU运行")
    
    # 初始化配置和模型
    config = SimpleConfig()
    model = SimpleModel(config)
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练模型
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 开始训练...")
    train_model(model, dataloader, optimizer, criterion)
    
    # 测试推理
    test_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    predictions = inference(model, test_input)
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - \n推理测试:")
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 输入: {test_input}")
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 预测: {predictions}")
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 真实值近似: [8.0, 18.0]")
    
    # 保存为Hugging Face格式
    save_dir = "./simple_model_hf"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - \n模型已保存到: {save_dir}")
    
    # 直接使用torch保存模型（备用方法）
    torch.save(model.state_dict(), "./simple_model.pth")
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 模型已保存为pth格式")
    
    # 保存为ONNX格式
    dummy_input = torch.randn(1, 2)  # 创建一个示例输入
    onnx_path = "./simple_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=["input"], 
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 模型已保存为ONNX格式: {onnx_path}")
    
    # 加载模型测试
    logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - \n加载保存的模型测试:")
    try:
        # 尝试使用Hugging Face格式加载
        loaded_model = SimpleModel.from_pretrained(save_dir)
        loaded_predictions = inference(loaded_model, test_input)
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - Hugging Face格式加载后预测: {loaded_predictions}")
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - Hugging Face格式模型加载成功!")
    except Exception as e:
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - Hugging Face格式加载失败: {e}")
        
        # 尝试使用torch格式加载
        logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - 尝试使用torch格式加载...")
        try:
            loaded_model = SimpleModel(config)
            loaded_model.load_state_dict(torch.load("./simple_model.pth"))
            loaded_predictions = inference(loaded_model, test_input)
            logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - torch格式加载后预测: {loaded_predictions}")
            logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - torch格式模型加载成功!")
        except Exception as e2:
            logger.info(f"{__file__}:{inspect.currentframe().f_lineno} - torch格式加载失败: {e2}")
            import traceback
            traceback.print_exc()
