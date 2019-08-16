# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multi-dSprites dataset reader."""

import functools
import tensorflow as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
    'binarized': 4,
    'colored_on_grayscale': 6,
    'colored_on_colored': 5
}
BYTE_FEATURES = ['mask', 'image']


def feature_descriptions(max_num_entities, is_grayscale=False):
  """Create a dictionary describing the dataset features.

  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks and generative factors returned per scene.
    is_grayscale: bool. Whether images are grayscale. Otherwise they're assumed
      to be RGB.

  Returns:
    A dictionary which maps feature names to `tf.Example`-compatible shape and
    data type descriptors.
  """

  num_channels = 1 if is_grayscale else 3
  return {
      'image': tf.FixedLenFeature(IMAGE_SIZE+[num_channels], tf.string),
      'mask': tf.FixedLenFeature(IMAGE_SIZE+[max_num_entities, 1], tf.string),
      'x': tf.FixedLenFeature([max_num_entities], tf.float32),
      'y': tf.FixedLenFeature([max_num_entities], tf.float32),
      'shape': tf.FixedLenFeature([max_num_entities], tf.float32),
      'color': tf.FixedLenFeature([max_num_entities, num_channels], tf.float32),
      'visibility': tf.FixedLenFeature([max_num_entities], tf.float32),
      'orientation': tf.FixedLenFeature([max_num_entities], tf.float32),
      'scale': tf.FixedLenFeature([max_num_entities], tf.float32),
  }


def _decode(example_proto, features):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  # To return masks in the canonical [entities, height, width, channels] format,
  # we need to transpose the tensor axes.
  single_example['mask'] = tf.transpose(single_example['mask'], [2, 0, 1, 3])
  return single_example


def dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
            map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    dataset_variant: str. One of ['binarized', 'colored_on_grayscale',
      'colored_on_colored']. This is used to identify the maximum number of
      entities in each scene. If an incorrect identifier is passed in, the
      TFRecords file will not be read correctly.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  if dataset_variant not in MAX_NUM_ENTITIES:
    raise ValueError('Invalid `dataset_variant` provided. The supported values'
                     ' are: {}'.format(list(MAX_NUM_ENTITIES.keys())))
  max_num_entities = MAX_NUM_ENTITIES[dataset_variant]
  is_grayscale = dataset_variant == 'binarized'
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  features = feature_descriptions(max_num_entities, is_grayscale)
  partial_decode_fn = functools.partial(_decode, features=features)
  return raw_dataset.map(partial_decode_fn,
                         num_parallel_calls=map_parallel_calls)
