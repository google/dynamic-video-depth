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

import os
from os.path import join
import time
from shutil import rmtree
from tqdm import tqdm
import torch
from options import options_test
import datasets
import models
from util.util_print import str_error, str_stage, str_verbose
import util.util_loadlib as loadlib
from loggers import loggers
from argparse import Namespace
print("Testing Pipeline")

###################################################

print(str_stage, "Parsing arguments")
opt = options_test.parse()
opt.full_logdir = None
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)

###################################################

print(str_stage, "Setting up output directory")
output_dir = opt.output_dir
output_dir += (opt.net + '_' + opt.dataset + '_' + opt.suffix.format(**vars(opt))) \
    if opt.suffix != '' else (opt.net + '_' + opt.dataset)
opt.output_dir = output_dir
if os.path.isdir(join(output_dir, 'epoch_%04d' % opt.epoch)):
    if opt.overwrite:
        rmtree(join(output_dir, 'epoch_%04d' % opt.epoch))
    else:
        raise ValueError(str_error + " %s already exists, but no overwrite flag"
                         % output_dir)
os.makedirs(output_dir, exist_ok=True)
opt.output_dir = output_dir

###################################################

print(str_stage, "Setting up loggers")
logger_list = [
    loggers.TerminateOnNaN(),
]
if opt.html_logger:
    html_summary_filepath = os.path.join(opt.output_dir, 'summary')
    html_logger = loggers.HtmlLogger(html_summary_filepath)
    logger_list.append(html_logger)
logger = loggers.ComposeLogger(logger_list)

###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net, test=True)
# load opt_original
opt_dict = torch.load(join(opt.checkpoint_path, 'opt.pt'))
opt_train = Namespace(**opt_dict)
opt_train.global_rank = 0
opt_train.output_dir = opt.output_dir
model = Model(opt_train, logger)
if hasattr(opt_train, 'midas'):
    opt.midas = opt_train.midas

# checkpoint_path
if opt.epoch < 0:
    net_file = join(opt.checkpoint_path, 'best.pt')
else:
    net_file = join(opt.checkpoint_path, 'nets', '%04d.pt' % opt.epoch)


###################################################
print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset(opt_train.dataset)
dataset = Dataset(opt_train, mode='vali', model=model)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=False,
    shuffle=False
)
n_batches = len(dataloader)
dataiter = iter(dataloader)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# test points: " + str(len(dataset)))
print(str_verbose, "# test batches: " + str(n_batches))

if hasattr(model, 'update_opt'):
    model.update_opt(opt_train, is_train=False)

model.load_state_dict(net_file)
model.to(device)
model.eval()
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))
###################################################

print(str_stage, "Testing")
if opt.html_logger:
    html_logger.on_train_begin()
    html_logger.training = False
    html_logger.on_epoch_begin(0)

model.opt.epoch = opt.epoch
for i in tqdm(range(n_batches)):
    batch = next(dataiter)
    model.test_on_batch(i, batch)

if hasattr(model, 'on_test_end'):
    model.on_test_end()
