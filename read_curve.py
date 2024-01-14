import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


path = r'E:\2023.6.18\3_out\dolp'
path_list = os.listdir(path)
path_list = sorted(path_list, key=lambda x:int(x.split('.')[0]))
x = np.arange(len(path_list))
cube = []
for i in range(len(path_list)):
    src = cv2.imread(os.path.join(path_list[i]), 0)
    cube.append(src)
cube = np.array(cube)
print(cube.shape)
plt.plot(x, cube[300, 300, :])
plt.show()