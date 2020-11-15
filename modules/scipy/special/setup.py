from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('special', parent_package, top_path)
    config.add_data_dir('tests')
    return config