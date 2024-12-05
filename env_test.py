import tensorflow as tf
import torch

def check_cuda_tensorflow():
    print("TensorFlow CUDA Check:")
    if tf.test.is_built_with_cuda():
        print("TensorFlow is built with CUDA")
    else:
        print("TensorFlow is not built with CUDA")

    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow GPU is available")
    else:
        print("TensorFlow GPU is not available")

def check_cuda_pytorch():
    print("\nPyTorch CUDA Check:")
    if torch.cuda.is_available():
        print("PyTorch CUDA is available")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch CUDA is not available")

if __name__ == "__main__":
    check_cuda_tensorflow()
    check_cuda_pytorch()