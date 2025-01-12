import tensorflow as tf
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
plt.plot([1,2,3,4])
plt.ylabel('some num')
plt.savefig("result")