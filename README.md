# 3D_synthesis_tool

## Environments
* Test with Python 3.7.6
* open3d 0.10.0

## Run
```
python3 synthesizer.py input_json output_folder num_of_synthetic_data
```
you can check the parameters information by
```
python3 synthesizer.py -h
```

## Output
```
output_path
├── annotations.json
├── depth/
    ├── xxx.npy
├── depth_images/
    ├── xxx.png
├── images/
    ├── xxx.jpg
├── point_cloud/
    ├── xxx.ply
```

## Visualize
```
python3 visualizer.py pcd_path
```

## Example
here is an example you can test directly:
```
python3 synthesizer.py 01_01_3d.json output/ 2
python3 visualizer.py output/01_01_3d/point_cloud/pcd_0.ply
```
