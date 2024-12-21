import torch
import numpy as np
import random
import os
import codecs
import logging
import builtins
import datetime
import torch.distributed as dist
from thop import profile
# import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel()/1e6 for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size,label_token=None,device='cpu'):
    input = torch.randn(*input_size).cuda()
    # flops, _= profile(model, inputs=(input,))
    flops, _ = profile(model, inputs=(input,label_token))
    return flops/1e9


def read_file(file_path):
    list = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按空格分割每行的内容
            parts = line.strip().split()
            if len(parts) >= 2:
                # 提取第二列的值
                list.append(parts[1])
    return list


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def compute_ACC(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def compute_MCACC(logits, labels):
    preds = torch.argmax(logits, dim=1)
    unique_labels = labels.unique()

    acc_per_class = []
    for label in unique_labels:
        correct = (preds == label)[labels == label].type_as(labels).float().mean().item()
        acc_per_class.append(correct)

    return sum(acc_per_class) / len(acc_per_class)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_log(args, record_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(record_path, 'recording.log')

    fh = logging.FileHandler(log_path, mode='w') 
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_models_to_half(model):
    for p in model.parameters():
        p.data = p.data.half()
        if p.grad:
            p.grad.data = p.grad.data.half()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def write_description_to_folder(file_name, config):
    with codecs.open(file_name, 'w') as desc_f:
        desc_f.write("- Training Parameters: \n")
        for key, value in config.__dict__.items():
            desc_f.write("  - {}: {}\n".format(key, value))
