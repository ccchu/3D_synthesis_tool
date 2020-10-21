import os
import cv2
import glob
import json
import copy # for deep copy usage
import random
import argparse

import numpy as np
import open3d as o3d

def vis_anno(img, anno):
    # Radius of circle
    radius = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    vis_img = img.copy()
    for i, kp in enumerate(anno["keypoints"].astype(int)):
        if kp[2] == 0: continue
        cv2.circle(vis_img, (kp[0], kp[1]), radius, color, thickness)
        cv2.putText(vis_img, f'{i}', (kp[0] - 3, kp[1] - 3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))

    x0, y0, w, h = anno["bbox"]
    vis_img = cv2.rectangle(vis_img, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), thickness)

    return vis_img

class JsonParser(object):
    def __init__(self, json_path):
        self.json_data = json.load(open(json_path, "r"))
        self.annos = []
        self.images = {}
        self._load_images()
        self._load_annos()

    def __load_depth(self, depth_value, width, height):
        DEPTH_INF_VALUE = 10000000000.0
        '''
        depth = np.array(depth_value)
        depth = np.where(depth == DEPTH_INF_VALUE, 0, depth)
        depth.resize(height, width)
        '''
        depth = np.load(depth_value) # depth_value: file path
        depth = np.where(depth == DEPTH_INF_VALUE, 0, depth)

        return depth

    def __load_kps(self, kps_list):
        kps = np.resize(np.array(kps_list), (len(kps_list)//3, 3))
        return kps

    def _load_images(self):
        for img in self.json_data["images"]:
            self.images[img["id"]] = cv2.imread(img["file_name"])

    def _load_annos(self):
        for anno in self.json_data["annotations"]:
            img_id = anno["image_id"]
            img = self.images[img_id]
            h, w, c = img.shape
            self.annos.append({
                "img_id": anno["image_id"],
                "bbox": anno["bbox"],
                "keypoints": self.__load_kps(anno["keypoints"]),
                "depth": self.__load_depth(anno["depth"], w, h),
                "keypoints_depth_real": anno["keypoints_depth_real"]
            })

    def datas(self):
        return self.images, self.annos, self.json_data


class Scene():
    def __init__(self, frame_path, d_path):
        self.image = cv2.imread(frame_path)
        self.pcd = o3d.io.read_point_cloud(d_path, remove_nan_points=False)
        self.points = np.array(self.pcd.points)
        self.depth = self.load_depth()
        self.depth_scale_img = None
        #self.load_depth()
        print(self.points.shape)
        print(self.depth.shape)

    def load_depth(self):
        DEPTH_INF_VALUE = 0
        DISTANCE_TO_PIXEL = 10 / 255
        depth = np.where(np.isnan(self.points[:, -1]), DEPTH_INF_VALUE, self.points[:, -1])
        depth = np.resize(depth, (480, 640))
        #cv2.imwrite("0929_depth.png", depth / DISTANCE_TO_PIXEL)

        return depth

    def depth_scale_image(self, width, height):
        if self.depth_scale_img is None:
            self.depth_scale_img = cv2.resize(self.image, (width, height))

        return self.depth_scale_img.copy()


class Synthesizer():
    def __init__(self, intrinsic, width, height, images, annos, scenes, json_data):
        self.intrinsic = intrinsic
        self.height = height
        self.width = width
        self.images = images
        self.annos = annos
        self.scenes = scenes # list of Scene
        self.json_data = json_data

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

    def output_depth_image(self, filename, img):
        DISTANCE_TO_PIXEL = 10 / 255  # meter / pixel
        cv2.imwrite(filename, 255 - img / DISTANCE_TO_PIXEL)

    def sample(self, n_samples, output_path):
        output_path = os.path.realpath(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path, "depth"))
            os.makedirs(os.path.join(output_path, "images"))
            os.makedirs(os.path.join(output_path, "point_cloud"))
            os.makedirs(os.path.join(output_path, "depth_images"))

        annotations = []
        images = []
        for i in range(n_samples):
            annos_idx = random.randint(0, len(annos) - 1)
            scenes_idx = 0
            target_depth = random.uniform(2, 3.6)
            anno = copy.deepcopy(self.annos[annos_idx])
            cnt_try = 0
            while True:
                try:
                    rgb_img, d_img, pcd= self._synthesize(anno, self.scenes[scenes_idx], target_depth)
                    break
                except Exception as e:
                    cnt_try += 1
                    if cnt_try > 10: break
                    print(f'{os.path.basename(output_path)}.json, annotation idx: {annos_idx}, target_depth: {target_depth}')
                    print(e)
            o3d.io.write_point_cloud(os.path.join(output_path, "point_cloud", f"pcd_{i}.ply"), pcd)

            #rgb_img = vis_anno(rgb_img, anno)
            #d_img = vis_anno(d_img, anno)

            image_id = f"image_{i}"
            image_filename = os.path.join(output_path, "images", f"{image_id}.jpg")
            depth_filename = os.path.join(output_path, "depth", f"depth_{i}.npy")
            cv2.imwrite(image_filename, rgb_img)
            np.save(depth_filename, d_img)

            #anno["depth"] = depth_filename
            _ = anno.pop("depth") # remove depth np array
            anno["id"] = image_id
            anno["img_id"] = image_id
            anno["keypoints"] = anno["keypoints"].flatten().tolist()
            #cv2.imwrite(os.path.join(output_path, f"image_{i}.jpg"), rgb_img)
            self.output_depth_image(os.path.join(output_path, "depth_images", f"depth_{i}.png"), d_img)

            annotations.append(anno)
            images.append({"file_name": image_filename,
                           "depth_name": depth_filename,
                           "id": image_id,
                           "width": 640,
                           "height": 320
                          })

        json_out = {'categories': json_data['categories']}
        json_out["annotations"] = annotations
        json_out["images"] = images

        with open(os.path.join(output_path, "annotations.json"), "w") as f:
            json.dump(json_out, f, indent=2)

    def _synthesize(self, anno, scene, traget_depth = 2.):
        x, y, w, h = anno["bbox"]
        rgb_obj = imgs[anno["img_id"]][y: y+h, x: x+w]
        d_obj = anno["depth"][y: y+h, x: x+w]
        fg = np.nonzero(d_obj)
        print(d_obj[fg], np.mean(d_obj[fg]))
        d_offset = traget_depth - np.mean(d_obj[fg])
        d_obj[fg] += d_offset
        #d_obj[fg] = d_obj[fg] - np.mean(d_obj[fg]) + traget_depth
        print(d_obj[fg], np.mean(d_obj[fg]))

        if scene.depth is None:
            scene.depth = self.point_cloud_to_depth(scene.pcd)
        d_scene = scene.depth.copy()

        pts = scene.points
        pts_target = pts[np.where(np.abs(pts[:, -1] - traget_depth) < 0.05)]
        #print(pts_target)

        uvd = np.zeros(pts_target.shape)
        for i, pt in enumerate(pts_target):
            tmp = np.matmul(self.intrinsic, pt)
            uvd[i, :] = tmp / tmp[-1]
        v_offset = int(round(np.mean(uvd[:, 1]) - np.amax(fg[0])))
        #print(np.amin(uvd[:, 0]), np.amax(uvd[:, 0]))
        u_offset = random.randint(int(round(np.amin(uvd[:, 0]))), 
                                  int(round(np.amax(uvd[:, 0]))) - w)

        print(u_offset, v_offset)
        depth_scale_img = scene.depth_scale_image(self.width, self.height)
        for i in range(fg[0].size):
            depth_scale_img[fg[0][i] + v_offset, fg[1][i] + u_offset] = rgb_obj[fg[0][i], fg[1][i]]
            d_scene[fg[0][i] + v_offset, fg[1][i] + u_offset] = d_obj[fg[0][i], fg[1][i]]

        pcd_out = self.depth_to_point_cloud(depth_scale_img, d_scene)

        # anno
        anno["bbox"] = [u_offset, v_offset, w, h]
        for i, kp in enumerate(anno["keypoints"]):
            if kp[2] == 0: continue
            kp[0] += u_offset - x
            kp[1] += v_offset - y
            anno["keypoints"][i] = kp
        anno["keypoints_depth_real"] = [d + d_offset for d in anno["keypoints_depth_real"]]
        anno["3D_keypoints"] = []
        i = 0
        for kp in anno["keypoints"].astype(int):
            if kp[2] == 0: continue
            anno["3D_keypoints"].append(self.uvd2xyz(kp[0], kp[1], anno["keypoints_depth_real"][i]))
            i+=1
        # o3d.io.write_point_cloud("test_0928.ply", pcd_out)
        # cv2.imwrite("0928.jpg", depth_scale_img)
        # self.output_depth_image("0928_depth.png", d_scene)
        return depth_scale_img, d_scene, pcd_out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="input file or folder")
    parser.add_argument("output_path", help="output folder")
    parser.add_argument("num_sythetic_data", type=int, help="number of output sythetic data")

    args = parser.parse_args()
    input_path = args.input_path
    if os.path.isdir(input_path):
        jsons = glob.glob(os.path.join(input_path, "*.json"))
    elif os.path.isfile(input_path):
        jsons = [input_path]
    else:
        print(f"Error format... {input}")

    output_prefix = args.output_path

    '''
    factor = 4.05
    intrinsic = np.array([[2.0029103327915384e+03, 0., 1.2630067590766187e+03],
                          [0., 1.9998826628301181e+03, 9.4133503449538910e+02],
                          [0., 0., 1. ]])
    intrinsic *= np.array([1 / factor, 1 / factor, 1])[:, np.newaxis]
    '''
    intrinsic = np.array([[5.0031619815080609e+02, 0., 3.1048676853792432e+02],
                          [0., 4.9968832903018671e+02, 2.3441460798378830e+02],
                          [0., 0., 1. ]])
    image_width = 640
    image_height = 480

    #jsons_path = "/mnt/upload/rocket/person/dataset/cmu/train/run0/json"
    #jsons = glob.glob(os.path.join(jsons_path, "*.json"))
    #output_prefix = "/mnt/upload/ccchu/3d_synthesize_data/dataset/"

    pcd_scene = Scene("r2_scene.jpg", "r2_scene.pcd")

    # continue
    exist_ids = glob.glob(os.path.join(output_prefix, "*"))
    exist_ids = [os.path.basename(d) for d in exist_ids]

    #imgs, annos, json_data = JsonParser("01_01_3d.json").datas()
    for json_path in jsons:
        output_folder = os.path.splitext(os.path.basename(json_path))[0]
        if output_folder in exist_ids:
            print(f"{output_folder} exists, skip...")
            continue

        imgs, annos, json_data = JsonParser(json_path).datas()
        print("load json done")
        print(len(annos))

        synthesizer = Synthesizer(intrinsic, image_width, image_height, imgs, annos, [pcd_scene], json_data)
        synthesizer.sample(args.num_sythetic_data, os.path.join(output_prefix, output_folder))
    #synthesizer.sample(3, "./output/0928")
    #synthesizer.sample(20, "/mnt/upload/ccchu/3d_synthesize_data/dataset/01_01_3d")

