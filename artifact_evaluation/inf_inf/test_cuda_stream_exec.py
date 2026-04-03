import argparse
import torch
import numpy as np
import time
import logging
from torchvision import models
from transformers import BertModel, BertConfig, BertTokenizer

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_and_input(model_name, batch_size, device):
    """
    根据模型名称加载预训练模型并生成模拟输入。

    Args:
        model_name (str): 要加载的模型的名称。
        batch_size (int): 输入数据的批量大小。
        device (torch.device): 模型和数据所在的设备。

    Returns:
        tuple: (模型实例, 模拟输入张量)
    """
    model_name = model_name.lower()
    logger.info(f"Loading model: {model_name} with batch size {batch_size}")

    if model_name in ['resnet50', 'resnet101', 'mobilenetv2']:
        if model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else: # mobilenetv2
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    elif model_name == 'bert':
        # 使用 Hugging Face Transformers 加载 BERT
        # 'bert-base-uncased' 是一个常见的选择
        model = BertModel.from_pretrained('bert-base-uncased')
        # BERT 通常需要 input_ids 和 attention_mask
        # 假设序列长度为 384
        seq_length = 384
        dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), device=device)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)
    model.eval() # 设置为评估模式
    return model, dummy_input

def main():
    """
    主函数，用于解析参数、加载模型并进行性能测试。
    """
    parser = argparse.ArgumentParser(description="Benchmark PyTorch model inference time.")
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        choices=['resnet50', 'resnet101', 'mobilenetv2', 'bert'],
        help='The model to benchmark.'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cuda:0',
        help='The device to run the model on (e.g., "cuda:0" or "cpu").'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=4,
        help='Batch size for the input data.'
    )
    parser.add_argument(
        '-w', '--warmup',
        type=int,
        default=10,
        help='Number of warm-up runs before measurement.'
    )
    parser.add_argument(
        '-r', '--repeat',
        type=int,
        default=100,
        help='Number of measurement runs.'
    )
    args = parser.parse_args()

    try:
        device = torch.device(args.device)
        if 'cuda' in args.device and not torch.cuda.is_available():
            logger.error(f"CUDA device '{args.device}' is not available. Please check your CUDA installation.")
            return
            
        # 1. 加载模型并准备输入数据
        model, dummy_input = get_model_and_input(args.model, args.batch_size, device)

    except Exception as e:
        logger.error(f"Failed to initialize model or data: {e}")
        return

    # 2. 预热
    if args.warmup > 0:
        logger.info(f"Warming up for {args.warmup} iterations...")
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(dummy_input)
        torch.cuda.synchronize(device)

    # 3. 重复执行并计时
    timings = []
    logger.info(f"Executing benchmark for {args.repeat} iterations...")
    with torch.no_grad():
        for _ in range(args.repeat):
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            _ = model(dummy_input)

            torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            
            timings.append((end_time - start_time) * 1000) # 转换为毫秒
    
    if not timings:
        logger.error("Execution failed or no timing data was collected.")
        return

    # 4. 统计并打印执行时间
    mean_time = np.mean(timings)
    std_dev = np.std(timings)
    p95 = np.percentile(timings, 95)
    p99 = np.percentile(timings, 99)

    logger.info(f"--- Benchmark Results for {args.model} (batch_size={args.batch_size}) ---")
    logger.info(f"Ran {args.repeat} times.")
    logger.info(f"Average latency: {mean_time:.4f} ms")
    logger.info(f"Standard Deviation: {std_dev:.4f} ms")
    logger.info(f"P95 Latency: {p95:.4f} ms")
    logger.info(f"P99 Latency: {p99:.4f} ms")
    logger.info("----------------------------------------------------")

if __name__ == "__main__":
    main()
