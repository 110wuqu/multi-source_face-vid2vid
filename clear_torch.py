import gc, torch
import shutil
from argparse import ArgumentParser
import os

dir_path = "/home/face-vid2vid-demo"

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def clear_log():
    log_dir = os.path.join(dir_path, 'log')
    logs = os.listdir(log_dir)
    for log in logs:
        each_log = os.path.join(log_dir, log)
        shutil.rmtree(each_log)
    print(os.listdir(log_dir))

def clear_tensorboard():
    tensorboard_dir = os.path.join(dir_path, 'tensorboard')
    tensorboards = os.listdir(tensorboard_dir)
    for tensorboard in tensorboards:
        each_tensorboard = os.path.join(tensorboard_dir, tensorboard)
        shutil.rmtree(each_tensorboard)
    print(os.listdir(tensorboard_dir))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--func", default="cache")
    opt = parser.parse_args()

    if opt.func == 'cache':
        clear_gpu()
    if opt.func == 'log':
        clear_log()
    if opt.func == 'tensorboard':
        clear_tensorboard()
        