import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_equal, assert_)
from pathlib import Path

from modules.experiment.autoencoder import file_basename, ckpt_nr


def test_filebasename():
    paths = ['file',
             u'file',
             'file.txt',
             u'file.txt',
             'file.tar.gz',
             'file.a.b.c.d.e.f.g',
             'relative/path/file.ext',
             '../relative/path/file.ext',
             '/absolute/path/file.ext',
             'Relative\\Windows\\Path\\file.txt',
             'C:\\Absolute\\Windows\\Path\\file.txt',
             '/path with spaces/file.ext',
             'C:\\Windows Path With Spaces\\file.txt',
             ]
    for path in paths:
        assert file_basename(path) == 'file'
    assert file_basename('some/path/file name with spaces.tar.gz.zip.rar.7z') == 'file name with spaces'


def test_ckpt_nr():
    path = Path('tmp')
    files = ['ckpt_10.index',
             'ckpt_20.index']
    path.mkdir(parents=True, exist_ok=True)
    for f in files:
        path.joinpath(f).touch()

    assert_array_equal(ckpt_nr(path), [10, 20])

    for f in files:
        path.joinpath(f).unlink()
    if not any(path.iterdir()):
        path.rmdir()
