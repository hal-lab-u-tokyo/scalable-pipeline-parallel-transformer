import os
import sys
import subprocess
import psutil
import torch


#実験の設定を行なって書き込める
def log_exp_config(logger, cfgs):
    """
    Log the experiment configuration to the console
    """
    logger.info("===== Experiment Configuration =====")
    logger.info("Working Directory: %s", os.getcwd())
    logger.info("Configs: %s", cfgs)


def log_sys_info(logging):
    """
    Log the system information to the console
    """
    logging.info("===== System Information =====")
    logging.info("Python Version: %s", sys.version)
    logging.info("PyTorch Version: %s", torch.__version__)
    logging.info("CUDA Version: %s", torch.version.cuda)
    logging.info("CuDNN Version: %s", torch.backends.cudnn.version())
    logging.info("NCCL Version: %s", torch.cuda.nccl.version())
    logging.info("CUDNN Benchmark: %s", torch.backends.cudnn.benchmark)
    logging.info("CUDNN Deterministic: %s", torch.backends.cudnn.deterministic)


def log_node_info(logger):
    """
    Log the node information to the console
    """
    logger.info("===== Node Information =====")
    logger.info("Host Name: %s", os.uname().nodename)
    logger.info("Total Memory: %.2f GB", psutil.virtual_memory().total / (1024**3))
    logger.info("Total CPU Cores: %d", os.cpu_count())
    logger.info("CPU Architecture: %s", os.uname().machine)
    logger.info("CUDA Device Count: %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        logger.info(
            "CUDA Device %d: %s, Compute Capability: %s, Memory: %.2f GB",
            i,
            torch.cuda.get_device_name(i),
            torch.cuda.get_device_capability(i),
            torch.cuda.get_device_properties(i).total_memory / (1024**3),
        )
    # Get GPU topology
    try:
        topologies = subprocess.run(
            ["nvidia-smi", "topo", "--matrix"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.info("GPU Topology: \n%s", topologies.stdout)
    except Exception as e:
        logger.error("Failed to get GPU topology: %s", e)


def log_gpu_static_info(logger, device):
    """
    Log the GPU information to the console
    """
    logger.info("===== GPU Information =====")
    logger.info("CUDA Device: %s", torch.cuda.get_device_name(device))
    logger.info("Compute Capability: %s", torch.cuda.get_device_capability(device))
    logger.info(
        "Memory: %.2f GB",
        torch.cuda.get_device_properties(device).total_memory / (1024**3),
    )


def log_gpu_dynamic_info(logger, device):
    """
    Log the GPU dynamic information to the console
    """
    raise NotImplementedError


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )


def log_model_info(logger, model):
    """
    Log the model information to the console
    """
    logger.info("===== Model Information =====")
    # logger.info("Model: %s", model)
    logger.info("Number of Parameters: %d", get_number_of_params(model))
    logger.info("Model Size: %.2f MB", get_model_size(model) / (1024**2))


def log_dataset_info(logger, dataset):
    """
    Log the dataset information to the console
    """
    raise NotImplementedError


def log_optimizer_info(logger, optimizer):
    """
    Log the optimizer information to the console
    """
    raise NotImplementedError


def log_scheduler_info(logger, scheduler):
    """
    Log the scheduler information to the console
    """
    raise NotImplementedError