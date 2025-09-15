import matplotlib.pyplot as plt
import numpy as np
from pypcd4 import PointCloud

if __name__ == "__main__":
    pc: PointCloud = PointCloud.from_path("/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_1/transit_20250909-133328_c300_id1328.pcd")

    array: np.ndarray = pc.numpy()
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.axis('equal')

    arr = array[array[:,0] > 3.3]     
    plt.scatter(arr[:,1], arr[:,2], c=arr[:,0], cmap='plasma', edgecolor='k', s=15)
    plt.xlabel('y'); plt.ylabel('z'); plt.colorbar(label='x value')
    plt.show()