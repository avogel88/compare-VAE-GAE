import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join, isdir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TensorEvent, ScalarEvent

__all__ = ['tabulate_events', 'tabulate_tensors', 'get_losses']
# ToDo: Interrupted training may cause empty logfiles in e.g. 'train' which leads to missing tag names.


def tabulate_events(path):
    """Retrieve wall_times, steps and values for given events in path.

    Concatenates values from different starts / different logs.
    """
    summary_iterators = [EventAccumulator(join(path, name)).Reload() for name in os.listdir(path)]
    tags = summary_iterators[0].Tags()['scalars']

    # for it in summary_iterators:
    #     assert it.Tags()['scalars'] == tags
    summary_iterators = [it for it in summary_iterators if it.Tags()['scalars'] == tags]

    out = defaultdict(list)
    steps = []
    wall_time = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        wall_time = [e.wall_time for e in summary_iterators[0].Scalars(tag)]

        for events in [acc.Scalars(tag) for acc in summary_iterators]:
        # for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            # assert len(set(e.step for e in events)) == 1

            out[tag] += [e.value for e in events]

    return wall_time, steps, out


#tr = '../../checkpoints/mnist/ae/0/train'
#print(pd.DataFrame(tabulate_events(tr)[-1]))


def tabulate_tensors(path):
    """Retrieve wall_times, steps and tensors for given events in path."""
    summary_iterators = [EventAccumulator(join(path, name)).Reload() for name in os.listdir(path)]
    tags = summary_iterators[0].Tags()['tensors']

    #for it in summary_iterators:
    #    assert it.Tags()['tensors'] == tags
    summary_iterators = [it for it in summary_iterators if it.Tags()['tensors'] == tags]

    #tags = set()
    #for it in summary_iterators:
    #    tags.update(it.Tags()['tensors'])

    out = defaultdict(list)
    steps = []
    wall_time = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Tensors(tag)]
        wall_time = [e.wall_time for e in summary_iterators[0].Tensors(tag)]

        for events in [acc.Tensors(tag) for acc in summary_iterators]:
        # for events in zip(*[acc.Tensors(tag) for acc in summary_iterators]):
            # assert len(set(e.step for e in events)) == 1

            out[tag] += [tf.make_ndarray(e.tensor_proto) for e in events]

    return wall_time, steps, out


def get_losses(path, subdirs=['train', 'test', 'validation']):
    """Retrieves the losses located in path + subdirectory."""
    df = pd.DataFrame()
    for sub in subdirs:
        if isdir(join(path, sub)):
            data = tabulate_events(join(path, sub))[-1]
            # tags, values = zip(*data.items())
            # tags = 'epoch_loss', 'epoch_accuracy'
            # values = 1D-list
            df[sub] = data['epoch_loss']
    return df


def get_images(path):
    """Retrieves the in tensorboard displayed images.

    Keeps only 10 samples in logs, format of samples is png (bytefield).
    Logs of previous runs are emptied.
    Thus tabulate_tensors fails, because the tags of previous runs are empty.
    """
    imgs = []
    wall_time, steps, data = tabulate_tensors(path)
    tags, values = zip(*data.items())
    for tag, tag_values in zip(tags, values):
        for step, samples in zip(steps, tag_values):
            # first 2 entries are x & y dimensions (not in byte format!)
            x = int(samples[0])
            y = int(samples[1])
            for sample in samples[2:]:
                # actual png in byte format (b'\x00...')
                np.frombuffer(sample, np.uint8)  # translates to uint8
                imgs += [sample]
    return imgs


# print(get_losses('../../checkpoints/gauss/vae/0/'))

# get correlation
# wall_time, steps, data = tabulate_tensors('../../checkpoints/gauss/vae/0/correlation')
# print(pd.DataFrame(data))

# get images
# get_images('../../checkpoints/mnist/vae/0/imgs')
