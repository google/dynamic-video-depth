# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
from datasets import get_dataset
from models import get_model


def add_general_arguments(parser):

    # GPU
    parser.add_argument('--gpu', type=str, required=True, help='gpu idx')

    # dataset
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset')

    # dataloader
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')

    # Network
    parser.add_argument('--net', type=str, required=True, help='name of the model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint path')
    parser.add_argument('--epoch', type=int, default=-1, help='epoch id for testing')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory")
    parser.add_argument('--overwrite', action='store_true',
                        help="Whether to overwrite the output folder if it exists")
    parser.add_argument('--suffix', default='epoch_{epoch}', type=str,
                        help="Suffix for `logdir` that will be formatted with `opt`, e.g., '{classes}_lr{lr}'")

    # visualizer
    parser.add_argument('--html_logger', action='store_true',
                        help="use html_logger for visualization")

    # Misc
    parser.add_argument('--manual_seed', type=int, default=None,
                        help='manual seed for randomness')

    return parser


def parse(add_additional_arguments=None):
    parser = argparse.ArgumentParser()
    parser = add_general_arguments(parser)
    if add_additional_arguments:
        parser, _ = add_additional_arguments(parser)
    opt_general, _ = parser.parse_known_args()
    net_name = opt_general.net

    dataset_name = opt_general.dataset
    # Add parsers depending on dataset and models
    parser, _ = get_dataset(dataset_name).add_arguments(parser)
    parser, _ = get_model(net_name).add_arguments(parser)

    # Manually add '-h' after adding all parser arguments
    if '--printhelp' in sys.argv:
        sys.argv.append('-h')

    opt = parser.parse_args()
    return opt
