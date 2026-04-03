import torch
import numpy as np
import time
import csv
import logging
from torchvision import models
from transformers import BertConfig, BertForQuestionAnswering

model_config =  {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_and_input(model_name, batch_size, device):
    """
    加载模型并生成模拟输入。

    Args:
        model_name (str): 模型名称
        batch_size (int): 批量大小
        device (torch.device): 设备

    Returns:
        tuple: (模型实例, 模拟输入张量或元组)
    """
    model_name = model_name.lower()
    logger.info(f"Loading model: {model_name} with batch size {batch_size}")

    if model_name in ['resnet50', 'resnet101', 'mobilenetv2']:
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:  # mobilenetv2
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    elif model_name == 'bert':
        config = BertConfig.from_dict(model_config)
        model = BertForQuestionAnswering(config).to(device)
        model.eval()
        # 利用 DummyDataLoader 生成模拟输入数据供推理使用（忽略 start/end positions）
        dummy_loader = DummyDataLoader(batch_size)
        input_ids, segment_ids, input_mask, _, _ = next(iter(dummy_loader))
        dummy_input = (input_ids.to(device), segment_ids.to(device), input_mask.to(device))

    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.to(device)
    model.eval()
    return model, dummy_input

def run_benchmark(model, dummy_input, warmup, repeat, device):
    """
    执行预热和基准测试，返回执行时延（毫秒）的列表。
    """
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)
        if 'cuda' in str(device):
            torch.cuda.synchronize(device)
    
    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            if 'cuda' in str(device):
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            if isinstance(dummy_input, tuple):
                _ = model(*dummy_input)
            else:
                _ = model(dummy_input)
            if 'cuda' in str(device):
                torch.cuda.synchronize(device)
            end_time = time.perf_counter()

            timings.append((end_time - start_time) * 1000)  # 转换为毫秒
    return timings

class DummyDataLoader():
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.input_ids = torch.ones((self.batchsize, 384), dtype=torch.int64)
        self.segment_ids = torch.ones((self.batchsize, 384), dtype=torch.int64)
        self.input_mask = torch.ones((self.batchsize, 384), dtype=torch.int64)
        self.start_positions = torch.zeros((self.batchsize,), dtype=torch.int64)
        self.end_positions = torch.ones((self.batchsize,), dtype=torch.int64)

    def __iter__(self):
        return self

    def __next__(self):
        return self.input_ids, self.segment_ids, self.input_mask, self.start_positions, self.end_positions

def main():
    # 如果有 CUDA 则使用 cuda:0，否则使用 cpu
    device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    # 定义要测试的模型和批量大小组合
    benchmarks = [
        # ('resnet50', 4),
        # ('resnet50', 32),
        # ('resnet101', 4),
        # ('resnet101', 32),
        # ('mobilenetv2', 4),
        # ('mobilenetv2', 32),
        ('bert', 4),
        ('bert', 32),
    ]
    
    # 参数配置
    warmup = 10
    repeat = 20
    results = []

    for model_name, batch_size in benchmarks:
        try:
            model, dummy_input = get_model_and_input(model_name, batch_size, device)
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            continue

        logger.info(f"开始测试: {model_name} 批量大小 {batch_size}")
        timings = run_benchmark(model, dummy_input, warmup, repeat, device)
        if not timings:
            logger.error(f"没有采集到时延数据: {model_name} - bs {batch_size}")
            continue

        avg_time = np.mean(timings)
        std_time = np.std(timings)
        p50 = np.percentile(timings, 50)
        p99 = np.percentile(timings, 99)

        results.append({
            'model': model_name.upper(),
            'method': 'pytorch',
            'bs': batch_size,
            'distribution': 'uniform',
            'avg': f"{avg_time:.4f}",
            'std': f"{std_time:.4f}",
            'p50': f"{p50:.4f}",
            'p99': f"{p99:.4f}"
        })
    
    csv_file = "benchmark_results.csv"
    logger.info(f"写入 CSV 文件: {csv_file}")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'method', 'bs', 'distribution', 'avg', 'std', 'p50', 'p99'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    logger.info("基准测试完成。")

if __name__ == "__main__":
    main()