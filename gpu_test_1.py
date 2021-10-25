#import os
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
print("Num GPUs Available: ", len(tf.test.gpu_device_name()))
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    print("Name:", gpu.name, "  Type:", gpu.device_type)

