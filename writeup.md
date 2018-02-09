## Project: Perception Pick & Place

---

[//]: # (Image References)

[pipeline1]: ./misc_imgs/pipeline1.png
[pipeline2]: ./misc_imgs/pipeline2.png
[pipeline3]: ./misc_imgs/pipeline3.png
[world1]: ./misc_imgs/world1.png
[world2]: ./misc_imgs/world2.png
[world3]: ./misc_imgs/world3.png

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

The pipeline of exercise 1 follows these steps:
Voxel down sampling -> filter passthrough z and y -> filter outlier removal -> RANSAC extract objects

```python
def pcl_callback(pcl_msg):
  ...
  # TODO: Voxel Grid Downsampling
  cloud_filtered = voxel_downsampling(cloud)
  # TODO: PassThrough Filter
  cloud_filtered = filter_passthrough_zy(cloud_filtered)
  # # TODO: Outlier Removal Filter
  cloud_filtered = filter_outlier_removal(cloud_filtered)
  # TODO: RANSAC Plane Segmentation
  cloud_objects = RANSAC_extract_objects(cloud_filtered)
  ...
```

![alt text][pipeline1]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

In this step, processed cloud is clustered using Euclidean and marked with color.

```python
...
# TODO: Euclidean Clustering
cluster_indices, white_cloud = cluster_objects_Euclidean(cloud_objects)
cluster_cloud = mark_objects_with_color(cluster_indices, white_cloud)
...
```

![alt text][pipeline2]

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

In this step, I used `hsv` color to compute color feature and increased the number of feature more than 1000 to improve the accuracy. I also increased the number of samples taken in `capture_features.py` to `150`.

* Compute histograms
```python
...
c1_hist = np.histogram(channel_1_vals, bins=50, range=(0, 256))
c2_hist = np.histogram(channel_2_vals, bins=50, range=(0, 256))
c3_hist = np.histogram(channel_3_vals, bins=50, range=(0, 256))
# TODO: Concatenate and normalize the histograms
hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0])).astype(np.float64)
normed_features = hist_features / np.sum(hist_features)
...
```

* Compute histograms of normal values
```python
...
nx_hist = np.histogram(norm_x_vals, bins=50, range=(0, 256))
ny_hist = np.histogram(norm_y_vals, bins=50, range=(0, 256))
nz_hist = np.histogram(norm_z_vals, bins=50, range=(0, 256)
# TODO: Concatenate and normalize the histograms
hist_features = np.concatenate((nx_hist[0], ny_hist[0], nz_hist[0])).astype(np.float64)
normed_features = hist_features / np.sum(hist_features)
...
```

* Result from trainning svm
![alt text][pipeline3]

```
Features in Training Set: 1350
Invalid Features in Training set: 2
Scores: [ 0.93703704  0.95185185  0.95555556  0.9330855   0.94423792]
Accuracy: 0.94 (+/- 0.02)
accuracy score: 0.944362017804
```

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

#### 2. Results

* ##### test1.world 100% (3/3)

![alt text][world1]

* ##### test2.world 80% (4/5)

I found this one can be occassionally confused and misclassified `glue` object 

![alt text][world2]

* ##### test3.world 87.5% (7/8)

This test also found to be occassionally confused and misclassified `glue` object 

![alt text][world3]

#### 3. Discussion

* It is important to map the color and normal features into feature vector following exactly the arrangement in trainning process.

* Increasing the number of sample improve significantly the accuracy, next is the number of features.

* There is a big trade off in down sampling step. More points mean prediction process has more material, however it will also increase the computation.

* I learn how to use `get_param` to read the param objects in `.yaml` and how to creat and assign value to ros messages.

* My implmentation still failed with `glue` object which I plan to resolve by increasing even more features and number of samples.




