import numpy as np
import open3d as o3d
from PIL import Image


def pcd_to_sideview_image(
    pcd_path,
    res=0.01,
    max_depth=0.7,
    normalize=True,
    binary=False,
    min_val=0,
    max_val=255,
):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Cloud is empty")

    Y,Z = pts[:,1], pts[:,2]
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()
    w = int(np.ceil((y_max - y_min) / res)) + 1
    h = int(np.ceil((z_max - z_min) / res)) + 1
    grid = np.zeros((h, w), dtype=np.uint32)
    y_idx = ((Y - y_min) / res).astype(np.int32)
    z_idx = ((z_max - Z) / res).astype(np.int32)
    np.add.at(grid, (z_idx, y_idx), 1)

    if binary:
        grid_img = np.where(grid > 0, max_val, min_val).astype(np.uint8)
    else:
        if normalize:
            if grid.max() == 0:
                grid_img = np.zeros_like(grid, dtype=np.uint8)
            else:
                grid_img = (grid / grid.max() * 255).astype(np.uint8)
        else:
            grid_img = np.clip(grid, 0, 255).astype(np.uint8)

    return Image.fromarray(grid_img, mode="L")


if __name__ == "__main__":
    img = pcd_to_sideview_image(
        pcd_path="/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_1/transit_20250909-133328_c300_id1328.pcd",
        res=0.04,
        max_depth=0.7,  # 70 cm depth
        normalize=True,
        binary=True,
    )
    img.save("vehicle_yz_first_70cm.png")
    img.show()
