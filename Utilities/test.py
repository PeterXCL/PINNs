import os

# Let TF see CUDA 12.3 DLLs
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin")

import tensorflow as tf

print("TF version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices('GPU'))