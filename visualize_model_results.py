import os
import cv2
import json
import argparse

import numpy as np
import open3d as o3d


def vis_3d_kps(kps, translation=0):
    points = np.array(kps)
    points[:, 0] += translation
    virtual1 = (points[1] + points[2]) /2
    virtual2 = (points[7] + points[8]) / 2 
    points = np.vstack([points, virtual1, virtual2])

    lines = np.array([[1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],
                      [0, 13], [13, 14], [1, 13], [2, 13], [7, 14], [8, 14]])
    line_colors  = np.array([[1, 0, 0], [1, 0, 0],
                             [1, 0, 0], [1, 0, 0],
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

class Converter():
    def __init__(self, intrinsic, width, height):
        self.intrinsic = intrinsic
        self.height = height
        self.width = width

    def point_cloud_to_depth(self, pcd):
        DISTANCE_TO_PIXEL = 10 / 255  # meter / pixel

        d_img = np.zeros((self.height, self.width), np.float32)
        for pt in np.array(pcd.points):
            u, v, d = np.matmul(self.intrinsic, pt)
            #d_img[int(v//d), int(u//d)] = d / DISTANCE_TO_PIXEL
            d_img[int(round(v / d)), int(round(u / d))] = d

        return d_img

    def depth_to_point_cloud(self, rgb, depth):
        fx = self.intrinsic[0, 0]
        cx = self.intrinsic[0, 2]
        fy = self.intrinsic[1, 1]
        cy = self.intrinsic[1, 2]
        #print(fx, fy, cx, cy)

        d_img = depth.copy()
        img_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        xyz = []
        color = []
        for u in range(self.width):
            for v in range(self.height):
                z = d_img[v, u]
                if z < 0.01:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                xyz.append([x, y, z])
                color.append((img_rgb[v, u] / 255).tolist())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
        pcd.colors = o3d.utility.Vector3dVector(np.array(color))

        return pcd

    def uvd2xyz(self, u, v, d):
        fx = self.intrinsic[0, 0]
        cx = self.intrinsic[0, 2]
        fy = self.intrinsic[1, 1]
        cy = self.intrinsic[1, 2]

        z = d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return [x, y, z]


if __name__ == '__main__':

    factor = 4.05
    intrinsic = np.array([[2.0029103327915384e+03, 0., 1.2630067590766187e+03],
                          [0., 1.9998826628301181e+03, 9.4133503449538910e+02],
                          [0., 0., 1. ]])
    intrinsic *= np.array([1 / factor, 1 / factor, 1])[:, np.newaxis]

    width = 640
    height = 480

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help='input json path')
    parser.add_argument("index", type=int, default=0, help='index of anno')

    args = parser.parse_args()

    json_data = json.load(open(args.input_path, "r"))
    anno = json_data["annotations"][args.index]
    images = json_data["images"]
    img_info = images[args.index]

    if img_info["id"] != anno["image_id"]:
        print(f"id not match, img: {img_info['id']} - anno: {anno['image_id']}, searching... ")
        for img in images:
            if img["id"] == anno["image_id"]:
                img_info = img
                break

    # load rgb 
    rgb = cv2.imread(img_info["file_name"])
    rgb_depth_scale = cv2.resize(rgb, (width, height))

    # load depth
    depth = np.load(img_info["depth_image"])
    DEPTH_INF_VALUE = 255
    depth = np.where(depth == DEPTH_INF_VALUE, 0, depth)

    # load pcd
    #pcd = o3d.io.read_point_cloud(img_info["pcd"])
    cvt = Converter(intrinsic, width, height)
    pcd = cvt.depth_to_point_cloud(rgb_depth_scale, depth)

    VOXEL_SIZE = 0.05
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    print(np.amax(np.array(pcd.points)[:,-1]))

    line_set = vis_3d_kps(anno["3Dkeypoints"])
    line_set1 = vis_3d_kps(anno["3Dkeypoints"], 1.5)
    line_set2 = vis_3d_kps(anno["3Dkeypoints"], -2.5)
    #o3d.visualization.RenderOption.line_width=10.0
    o3d.visualization.draw_geometries([pcd, line_set, line_set1, line_set2])
    #o3d.visualization.draw_geometries([line_set])
