"""CLEVR (with masks) dataset reader."""

import tensorflow as tf


COMPRESSION_TYPE = 'GZIP'
IMAGE_SIZE = [240, 320]
MAX_NUM_ENTITIES = 11  # The maximum number of foreground and background
                       # entities in the provided dataset. This corresponds to
                       # the number of segmentation masks returned per scene.


# Create a dictionary describing the dataset features.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in ['mask', 'image', 'color', 'material', 'shape', 'size']:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
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
  parsed_dataset = raw_dataset.map(_decode,
                                   num_parallel_calls=map_parallel_calls)
  return parsed_dataset
