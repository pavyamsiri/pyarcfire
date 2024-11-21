# PyArcFiRe

PyArcFiRe is a python port of [SpArcFiRe](https://github.com/waynebhayes/SpArcFiRe) which is written primarily in MatLab.
Like SpArcFiRe it can be used to detect spiral arcs in images, mostly for galaxy images however it perhaps may work in other contexts.

## Limitations

Note that this is currently a work in progress and the project may change greatly over time.

### Functionality

This port does not have all of the functionality and features of SpArcFiRe such as bar finding and fitting, automatic centering and deprojection, etc.

## Installation

You can install this package by simply using the command

```
$ pip install pyarcfire
```

## Interface

There are two main ways of using PyArcFiRe

1. As a python package to use in your own programs.
2. As a command line interface.

### Package

Create an instance of `SpiralFinder` and then call the `extract` method on a 2D array to run the spiral finding algorithm.

```python
from pyarcfire import SpiralFinder
import numpy as np

# Example: Load a grayscale image
image = ... # Replace with an actual 2D image array

# Create a SpiralFinder instance
finder = SpiralFinder()

# Extract spiral features
result = finder.extract(image)
```

Then to extract the spiral arm masks you can simply use the `mask` property to get a 2D array of integers. The non-zero values of the `mask` array is the cluster id of
the pixel.

For example, you can plot just the first cluster like this

```python
import matplotlib.pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(111)
axes.imshow(result.mask == 1)
plt.show()
plt.close(fig)
```

The original image and the preprocessed image can be accessed as well.

```python
import matplotlib.pyplot as plt

fig = plt.figure()
original_axes = fig.add_subplot(121)
original_axes.imshow(result.original_image)
processed_axes = fig.add_subplot(122)
processed_axes.imshow(result.processed_image)
plt.show()
plt.close(fig)
```

Additionally, you can get the dominant chirality and overall pitch angle of the result like so

```python
print(result.get_dominant_chirality()) # Chirality.NONE | Chirality.CLOCKWISE | Chirality.COUNTER_CLOCKWISE
print(result.get_overall_pitch_angle()) # +- n degrees
```

or get the log spiral fits to each cluster and plot them like so
```python
import matplotlib.pyplot as plt

fig = plt.figure()
axes = fig.add_subplot(111)

for cluster_index in range(result.num_clusters):
    fit = result.get_fit(cluster_index)
    x, y = spiral_fit.calculate_cartesian_coordinates(100, pixel_to_distance=1, flip_y=False) # `pixel_to_distance` converts from pixel units to your desired distance units
    axes.plot(x, y)
plt.show()
plt.close()
```

#### Algorithm Parameters

The spiral finding algorithm has many configurable parameters for each step.

When computing the orientation field the parameters are:

- `neighbour_distance (int)`: The distance in pixels between a cell and its neighbour when denoising the orientation field.
- `kernel_radius (int)`: The radius of the orientation filter kernel in pixels.
- `num_levels (int)`: The number of image rescalings to create orientation fields of and then join.

When computing the similarity matrix:

- `similarity_cutoff (float)`: The minimum allowed similarity between orientation field pixels.

When clustering pixels:

- `error_ratio_threshold (float)`: The maximum error ratio allowed for a merge between two clusters to be permitted.
- `merge_check_mininum_cluster_size (int)`: The maximum size of a cluster before merges with other clusters become checked.
- `minimum_cluster_size (int)`: The minimum cluster size allowed after all merges are completed.
- `remove_central_cluster (bool)`: A flag to remove remove the clusters that touch the center of the image.

When merging nearby clusters by checking their log spiral fits:

- `stop_threshold (float)`: The maximum merge error ratio before stopping merges.

These parameters can be configured by calling the associated methods on `SpiralFinder`

```python
from pyarcfire import SpiralFinder


# Create a SpiralFinder instance
finder = SpiralFinder()

# Adjust orientation field generation
finder = finder.with_orientation_field_settings(
    neighbour_distance=4,
    kernel_radius=None, # Set to `None` to keep the old value.
    # num_levels=3, # Omit the parameter if you want to keep the old value as well.
)

# Adjust similarity matrix generation
finder = finder.with_similarity_matrix_settings(
    cutoff=0.15,
)

# Adjust clustering
finder = finder.with_clustering_settings(
    error_ratio_threshold=2.5,
    merge_check_minimum_cluster_size=25,
    minimum_cluster_size=120,
    remove_central_cluster=False,
)

# Adjust merging by fit
finder = finder.with_merge_fit_settings(
    stop_threshold=2.4,
)
```

#### Preprocessing

In order for an image to be ran through the algorithm, some preprocessing may be necessary. The requirements are that:

- The array is 2D.
- The array values are normalized in the range [0, 1].
- The array's height and width must be divisible by 2^N where N is the number of orientation field levels.

Therefore `SpiralFinder` will by default perform the following:

1. It will first check that the array is 2D. If not an exception will be raised.
2. It will normalize the values in the array using a linear scale.
3. It will resize the image so that the height and width are valid by finding the closest valid size.

Also as part of the algorithm, a contrast boosting step will be performed. By default this is an unsharp mask.

These preprocessing steps can be changed however and in fact can be turned entirely off.

```python
from pyarcfire import SpiralFinder
from pyarcfire.preprocess import ImageIdentityNormalizer

# Turn off all preprocessing
finder = SpiralFinder().with_normalizer(None).with_resizer(None).with_booster(None)

# Change the normalizer
finder = SpiralFinder().with_normalizer(ImageIdentityNormalizer())
```

You can create custom preprocessors as well by implementing the `ImageNormalizer`, `ImageResizer` and `ImageContrastBooster` protocols (see `pyarcfire.preprocess`).


### Command Line Interface

PyArcFiRe can also be interacted with through the command line interface via `python -m pyarcfire ...`. Currently this is a work in progress and is mainly
a way to drive debugging code.
