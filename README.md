This repository contains the following datasets for multi-object representation
learning, used in developing scene decomposition methods like
[MONet](https://arxiv.org/abs/1901.11390) [1] and
[IODINE](http://proceedings.mlr.press/v97/greff19a.html) [2]:

1.  Multi-dSprites
2.  Objects Room
3.  CLEVR (with masks)
4.  Tetris

<p align="center">
  <img src="preview.png" width=546 height= 564>
</p>

The datasets consist of multi-object scenes. Each image is accompanied by
ground-truth segmentation masks for all objects in the scene. We also provide
per-object generative factors (except in Objects Room) to facilitate
representation learning.

Lastly, the `segmentation_metrics` module contains a TensorFlow implementation
of the
[Adjusted Rand index](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index),
which can be used to compare inferred object segmentations with ground-truth
segmentation masks.

## Bibtex

If you use one of these datasets in your work, please cite it as follows:

```
@misc{multiobjectdatasets19,
  title={Multi-Object Datasets},
  author={Kabra, Rishabh and Burgess, Chris and Matthey, Loic and
          Kaufman, Raphael Lopez and Greff, Klaus and Reynolds, Malcolm and
          Lerchner, Alexander},
  howpublished={https://github.com/deepmind/multi-object-datasets/},
  year={2019}
}
```

## Descriptions

### Multi-dSprites

This is a dataset based on
[dSprites](https://github.com/deepmind/dsprites-dataset). Each image consists of
multiple oval, heart, or square-shaped sprites (with some occlusions) set
against a uniformly colored background.

We're releasing three versions of this dataset containing 1M datapoints each:

1.1 Binarized: each image has 2-3 white sprites on a black background.

1.2 Colored sprites on grayscale: each scene has 2-5 randomly colored HSV
sprites on a randomly sampled grayscale background.

1.3 Colored sprites and background: each scene has 1-4 sprites. All colors are
randomly sampled RGB values.

Each datapoint contains an image, a number of background and object masks, and
the following ground-truth features per object: `x` and `y` positions, `shape`,
`color` (rgb values), `orientation`, and `scale`. Lastly, `visibility` is a
binary feature indicating which objects are not null.

### Objects Room

This dataset is based on the Mujoco environment used by the Generative Query
Network (Eslami et al. 2018) and is a multi-object extension of the
[3d-shapes dataset](https://github.com/deepmind/3d-shapes). The training set
contains 1M scenes with up to three objects. We also provide ~1K test examples
for the following variants:

2.1 Empty room: scenes consist of the sky, walls, and floor only.

2.2 Six objects: exactly 6 objects are visible in each image.

2.3 Identical color: 4-6 objects are placed in the room and have an identical,
randomly sampled color.

Datapoints consist of an image and fixed number of masks. The first four masks
correspond to the sky, floor, and two halves of the wall respectively. The
remaining masks correspond to the foreground objects.

### CLEVR

We adapted the
[open-source script](https://github.com/facebookresearch/clevr-dataset-gen)
provided by Johnson et al. 2017 to produce ground-truth segmentation masks for
all objects in the CLEVR scene. We ignore the original question-answering task.

The images and masks in the dataset are of size 320x240. We also provide all
ground-truth factors included in the original dataset (namely `x`, `y`, and `z`
position, `pixel_coords`, and `rotation`, which are real-valued; plus `size`,
`material`, `shape`, and `color`, which are encoded as integers) along with a
`visibility` vector to indicate which objects are not null.

### Tetris

Each 35x35 image contains three tetraminos, sampled from 17 unique
shapes/orientations. Each tetramino has one of six possible colors (red, green,
blue, yellow, magenta, cyan). We provide `x` and `y` position, `shape`, and
`color` (integer-coded) as ground-truth features. Datapoints also include a
`visibility` vector.

## Download

The datasets can be downloaded from
[Google Cloud Storage](https://console.cloud.google.com/storage/browser/multi-object-datasets).
Each dataset is a single TFRecords file.

## Usage

After downloading the dataset files, you can read them as `tf.data.Dataset`
instances with the readers provided. The example below shows how to read the
colored-sprites-and-background version of Multi-dSprites:

```
  import tensorflow as tf

  tf_records_path = 'path/to/multi_dsprites_colored_on_colored.tfrecords'
  batch_size = 32

  dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
  batched_dataset = dataset.batch(batch_size)  # optional batching
  iterator = batched_dataset.make_one_shot_iterator()
  data = iterator.get_next()

  with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
```

You can compare predicted object segmentation masks with the ground-truth masks
using `segmentation_metrics.adjusted_rand_index` as below:

```
  max_num_entities = multi_dsprites.MAX_NUM_ENTITIES_DICT['colored_on_colored']
  desired_shape = [batch_size,
                   multi_dsprites.IMAGE_SIZE[0] * multi_dsprites.IMAGE_SIZE[1],
                   max_num_entities]
  true_groups_oh = tf.transpose(data['mask'], [0, 2, 3, 4, 1])
  true_groups_oh = tf.reshape(true_groups_oh, desired_shape)
  true_groups_oh = tf.cast(true_groups_oh, tf.float32)

  random_prediction = tf.random_uniform(desired_shape[:-1],
                                        minval=0, maxval=max_num_entities,
                                        dtype=tf.int32)
  random_prediction_oh = tf.one_hot(random_prediction, depth=max_num_entities)

  ari = segmentation_metrics.adjusted_rand_index(true_groups_oh,
                                                 random_prediction_oh)
```

To exclude all background pixels from the ARI score (as in [2]), you can compute
it as follows instead. This assumes the first true group contains all background
pixels:

```
  ari_nobg = segmentation_metrics.adjusted_rand_index(true_groups_oh[..., 1:],
                                                      random_prediction_oh)
```

## References

[1] Burgess, C. P., Matthey, L., Watters, N., Kabra, R., Higgins, I., Botvinick,
M., & Lerchner, A. (2019). Monet: Unsupervised scene decomposition and
representation. arXiv preprint arXiv:1901.11390.

[2] Greff, K., Kaufman, R.L., Kabra, R., Watters, N., Burgess, C., Zoran, D.,
Matthey, L., Botvinick, M. & Lerchner, A. (2019). Multi-Object Representation
Learning with Iterative Variational Inference. Proceedings of the 36th
International Conference on Machine Learning, in PMLR 97:2424-2433.

## Disclaimers

This is not an official Google product.
