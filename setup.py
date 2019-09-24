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
"""Installation script for setuptools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup


EXTRA_PACKAGES = {
    'tf': ['tensorflow>=1.14'],
    'tf_gpu': ['tensorflow-gpu>=1.14'],
}

setup(
    name='multi_object_datasets',
    version='1.0.0',
    author='DeepMind',
    license='Apache License, Version 2.0',
    description=('Multi-object image datasets with'
                 'ground-truth segmentation masks and generative factors.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=['datasets', 'machine learning', 'representation learning'],
    url='https://github.com/deepmind/multi_object_datasets',
    packages=['multi_object_datasets'],
    package_dir={'multi_object_datasets': '.'},
    extras_require=EXTRA_PACKAGES,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
