# Copyright 2026 The multi_object_datasets Authors.
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
"""Tests for foreground-only adjusted Rand index scores."""

import tensorflow.compat.v1 as tf

from segmentation_metrics import adjusted_rand_index


class AdjustedRandIndexTest(tf.test.TestCase):

  def test_ignores_unlabeled_points_in_degenerate_partitions(self):
    # A negative index produces an all-zero ground-truth vector.
    true_labels = [[-1, 1, 1, -1], [-1, 0, 1, -1], [-1, 1, -1, -1],
                   [-1, 1, 1, -1]]
    pred_labels = [[0, 2, 2, 1], [0, 2, 1, 2], [0, 2, 1, 2],
                   [0, 1, 2, 1]]
    scores = adjusted_rand_index(tf.one_hot(true_labels, 3),
                                 tf.one_hot(pred_labels, 3))
    self.assertAllClose(self.evaluate(scores), [1., 1., 1., 0.])


if __name__ == '__main__':
  tf.test.main()
