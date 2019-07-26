"""Objects Room dataset reader."""

import functools
import tensorflow as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
MAX_NUM_ENTITIES_DICT = {
    # The maximum number of foreground and background entities in each variant
    # of the provided datasets. The values correspond to the number of
    # segmentation masks returned per scene.
    'train': 7,
    'six_objects': 10,
    'empty_room': 4,
    'identical_color': 10
}


def feature_descriptions(max_num_entities):
  """Create a dictionary describing the dataset features.

  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks returned per scene.

  Returns:
    A dictionary of feature descriptions.
  """
  features = {
      'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
      'mask': tf.FixedLenFeature([max_num_entities]+IMAGE_SIZE+[1], tf.string),
  }
  return features


def _decode(example_proto, features):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.parse_single_example(example_proto, features)
  for k in ['mask', 'image']:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
            map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    dataset_variant: str. One of ['train', 'six_objects', 'empty_room',
      'identical_color']. This is used to identify the maximum number of
      entities in each scene. If an incorrect identifier is passed in, the
      TFRecords file will not be read correctly.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  if dataset_variant not in MAX_NUM_ENTITIES_DICT:
    raise ValueError('Invalid `dataset_variant` provided. The supported values'
                     ' are: {}'.format(MAX_NUM_ENTITIES_DICT.keys()))
  max_num_entities = MAX_NUM_ENTITIES_DICT[dataset_variant]
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  features = feature_descriptions(max_num_entities)
  partial_decode_fn = functools.partial(_decode, features=features)
  parsed_dataset = raw_dataset.map(partial_decode_fn,
                                   num_parallel_calls=map_parallel_calls)
  return parsed_dataset
