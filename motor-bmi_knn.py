import numpy as np
import matplotlib as plt

def read_file(file_name):
    lines = [line.rstrip('\n').split('\t') for line in open(file_name)]
    file_np = np.array(lines).astype(np.float)
    if file_np.shape[0] == 1:
        file_np = file_np.flatten()
    return file_np

angle_training = read_file('Angle_Training.txt')
spike_training = read_file('Training_SpikeTrains.txt')
print(spike_training.shape)
