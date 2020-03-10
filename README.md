# Incremental SfM

## ABOUT
this project implements simple incremental sfm with the following functions.
- initialize
- feature matching
- solve pnp and incrementally add points
- loop detection 
- visualization (both point clouds and camera poses)

### future works
- loop closure
- bundle adjustment

### DEPENDENCIES
- OpenCV
 - including viz (for visualization)
- DVoW2 (for loop detection)
## HOW TO USE

### BUILD
```bash
mkdir build
cd build
cmake ..
make
```
### RUN 
- `setting.yml` should be at the same directory as executable. content should be as follows.
```yaml
%YAML:1.0
rootDir: <path_of_dataset> 
showMatches: false
showCameras: false
minHessian: 800
displayPcdCycle: 40
startFrame: 0
matchingRatioThresh: 0.7
errorRatio: 0.1
reprojectionErrorThresh: 0.4
```
- run the following command
```bash
./sfm
```

### TARGET DATA STRUCTURE
#### `images.txt`
- list of images (required)
```
<num_images>
<image_path_1>
<image_path_2>
...
<image_path_num_images>
```
#### `K.txt`
- intrinsic matrix (required)
```
fx 0 cx 0 fy cy 0 0 1
```
#### `distortion.txt`
- distortion params in opencv format (optional)
```
k1 k2 p1 p2 k3
```
## RESULTS

### temple dataset
![temple1](/results/temple1.png)
![temple2](/results/temple2.png)
### Kitti dataset
![kitti1](/results/kitti1.png)
![kitti2](/results/kitti2.png)

### gerrard hall dataset 
![hall1](/results/hall1.png)
![hall2](/results/hall2.png)
