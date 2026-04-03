from benchmarks.vision_models import vision
from benchmarks.conv import conv_loop
from benchmarks.transformer import transformer
from benchmarks.bert import bert
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, required=True,
                        help='profiler type, ncu or sys')
    parser.add_argument('--model_name', type=str, default='',
                        help='name of vision model to profile (leave empty for conv_loop)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size') 
    args = parser.parse_args()

    model_name = args.model_name
    profile_type = args.profile
    batch_size = args.batch_size

    if profile_type not in ['ncu', 'nsys']:
        print("error: profile unmatch.")
        exit(1)

    if model_name:  # 如果 model_name 不为空
        if model_name == 'bert':
            bert(batch_size, 0, True, profile_type)
        elif model_name == 'transformer':
            transformer(batch_size, 0, True, profile_type)
        else:
            vision(model_name, batch_size, 0, True, profile_type)
    else:  # model_name 为空，使用 conv_loop
        conv_loop(4, 0, True, profile_type)
