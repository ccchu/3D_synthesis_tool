import os
import cv2
import json
import argparse

import numpy as np
import open3d as o3d


def vis_3d_kps(kps):
    points = np.array(kps)
    virtual1 = (points[1] + points[2]) /2
    virtual2 = (points[7] + points[8]) / 2 
    points = np.vstack([points, virtual1, virtual2])

    lines = np.array([[1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],
                      [0, 13], [13, 14], [1, 13], [2, 13], [7, 14], [8, 14]])
    line_colors  = np.array([[1, 0, 0], [1, 0, 0],
                             [1, 1, 0], [1, 1, 0],
                             [0, 1, 0], [0, 1, 0],
                             [0, 1, 1], [0, 1, 1],
                             [0, 0, 1], [0, 0, 1],
                             [0, 0, 1], [0, 0, 1],
                             [0, 0, 1], [0, 0, 1]])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    return line_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help='input pcd path')

    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input_path)
    #pcd = o3d.io.read_point_cloud("/code/dump_bg_0427/20200427070445732.pcd")
    # print(np.array(pcd.points))
    # print(np.array(pcd.colors))
    arr = np.array(pcd.points)
    # print(np.amin(arr[:, 2]), np.amax(arr[:, 2]))

    dirname = os.path.dirname(args.input_path)
    filename = os.path.basename(args.input_path)
    id = os.path.splitext(filename)[0]
    id = id.replace("pcd", "image")
    print(f'id: {id}')

    annos = json.load(open(os.path.join(dirname, "../annotations.json"), "r"))
    for anno in annos["annotations"]:
        if anno["id"] == id:
            line_set = vis_3d_kps(anno["3D_keypoints"])

            #o3d.visualization.RenderOption.line_width=10.0
            o3d.visualization.draw_geometries([pcd, line_set])
