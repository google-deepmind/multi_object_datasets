# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""CATER (with masks) dataset reader."""

import functools
import tensorflow as tf


COMPRESSION_TYPE = 'ZLIB'
IMAGE_SIZE = [64, 64]
SEQUENCE_LENGTH = 33
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['image', 'mask']


def feature_descriptions(
    sequence_length=SEQUENCE_LENGTH,
    max_num_entities=MAX_NUM_ENTITIES):
  return {
      'camera_matrix': tf.io.FixedLenFeature(
          [sequence_length, 4, 4], tf.float32),
      'image': tf.io.FixedLenFeature([], tf.string),
      'mask': tf.io.FixedLenFeature([], tf.string),
      'object_positions': tf.io.FixedLenFeature(
          [max_num_entities, sequence_length, 3], tf.float32)
  }


def _decode(example_proto, features,
            sequence_length=SEQUENCE_LENGTH,
            max_num_entities=MAX_NUM_ENTITIES):
  """Parse the input `tf.Example` proto using a feature description dictionary.

  Args:
    example_proto: the serialized example.
    features: feature descriptions to deserialize `example_proto`.
    sequence_length: the length of each video in timesteps.
    max_num_entities: the maximum number of entities in any frame of the video.

  Returns:
    A dict containing the following tensors:
      - 'image': a sequence of RGB frames.
      - 'mask': a mask for all entities in each frame.
      - 'camera_matrix': a 4x4 matrix describing the camera pose in each frame.
      - 'object_positions': 3D position for all entities in each frame.
  """
  single_example = tf.io.parse_single_example(example_proto, features=features)

  for key in BYTE_FEATURES:
    single_example[key] = tf.io.decode_raw(single_example[key], tf.uint8)
  single_example['image'] = tf.reshape(
      single_example['image'],
      [sequence_length] + IMAGE_SIZE + [3])
  single_example['mask'] = tf.reshape(
      single_example['mask'],
      [sequence_length, max_num_entities] + IMAGE_SIZE + [1])
  single_example['object_positions'] = tf.transpose(
      single_example['object_positions'], [1, 0, 2])
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse TFRecords.

  Args:
    tfrecords_path: str or Sequence[str]. Path or paths to dataset files.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  features = feature_descriptions()
  partial_decode_fn = functools.partial(_decode, features=features)
  return raw_dataset.map(
      partial_decode_fn, num_parallel_calls=map_parallel_calls)
