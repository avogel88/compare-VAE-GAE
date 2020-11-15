from os.path import exists
import numpy as np
from numpy.random import randn, randint, seed
from scipy.stats import norm, exponweib
from modules.experiment import *
from modules.scipy.stats import empiric
from modules.tensorflow.keras.datasets import gaussian_mixture_generate
from tensorboard import program
import webbrowser


# start tensorboard
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'checkpoints'])
# url = tb.launch()
# webbrowser.open(url)

# plot empirical distributions of MNIST and Fashion-MNIST
plot_mnist_dist()
plot_alt_dist()

# plot exemplary marginal gaussianization
# ToDo: plot_marginal_gauss causes error in gaussian_mixture_generate
if not exists(config('PATHS')['marginal_gauss']):
    x1 = np.arange(0, 3, .01)
    x2 = np.arange(-3, 3, .01)
    pdf = exponweib.pdf(x1, 2, 2) + randn(len(x1)) * norm.pdf(x1, loc=1, scale=.5) / 20
    pdf = (pdf - min(pdf))
    pdf /= np.sum(np.abs(pdf))
    emp = empiric(values=(x1, pdf), reconstruct=True)
    plot_marginal_gauss((x1, x2), (emp, norm), 1.5, config('PATHS')['marginal_gauss'])

# prepare data
file = config('DATA')['normal_samples']
if not exists(file):
    seed(randint(10 ** 9))
    prepare_z(file, samples=60000, dim=10)
file = config('DATA')['gaussian_mixture']
if not exists(file):
    seed(randint(10 ** 9))
    # ToDo: numpy.linalg.LinAlgError: SVD did not converge
    # Reason: plot_marginal_gauss
    # Solution: Restart script, plot_marginal_gauss is not executed once the file exists
    gaussian_mixture_generate(file, train=60000, test=60000, validate=10000)

# training
continued_training('QUANT')
continued_training('QUAL')

quant = config('QUANT')
qual = config('QUAL')
quant_runs = int(quant['runs'])
quant_epochs = int(quant['epochs'])
quant_ksepochs = int(quant['ksepochs'])
qual_runs = int(qual['runs'])
qual_epochs = int(qual['epochs'])
qual_ksepochs = int(qual['ksepochs'])

# plotting metrics
for model in quant['models'].split(','):
    file = config('PATHS', dataset='gauss', model=model)['hist_img']
    if not exists(file):
        plot_history('gauss', model, quant_runs, quant_epochs)
for model in qual['models'].split(','):
    for dataset in qual['datasets'].split(','):
        file = config('PATHS', dataset=dataset, model=model)['hist_img']
        if not exists(file):
            plot_history(dataset, model, qual_runs, qual_epochs)

# plot ks statistics
period = int(config('GENERAL')['save_period'])
file_mnist = config('PATHS', dataset='mnist', model='gae', epoch=qual_epochs)['kse_img']
file_fashion = config('PATHS', dataset='fashion', model='gae', epoch=qual_epochs)['kse_img']
file_gauss = config('PATHS', dataset='gauss', model='gae', epoch=quant_epochs)['kse_img']
if not exists(file_mnist):
    plot_ks('mnist', qual_runs, qual_epochs-period)
if not exists(file_fashion):
    plot_ks('fashion', qual_runs, qual_epochs-period)
if not exists(file_gauss):
    plot_ks('gauss', quant_runs, quant_epochs-period)

# QUALITATIVE: sampling and plotting, ROC curves for anomaly detection
# ToDo: RBIG.rot is all NaN in first epoch.
# Reason: plot_history, plot_ks
# Solution: Execute parts in terminal
seednr = 12345
plot_anomalies(
    models=['ae', 'vae', 'gae'],
    orig='mnist',
    anomaly='fashion',
    path=config('PATHS')['anomaly_mnist'],
    seed=seednr)
plot_anomalies(
    models=['ae', 'vae', 'gae'],
    orig='fashion',
    anomaly='mnist',
    path=config('PATHS')['anomaly_fashion'],
    seed=seednr)
plot_comparison(seed=seednr)

# time series in logarithmic scale
# ToDo: RBIG.rot is all NaN in first epoch.
# Reason: plot_history, plot_ks
# Solution: Execute parts in terminal
max_epochs = int(config('QUAL')['epochs'])
major_ticks = 10 ** np.arange(1, 5)
minor_ticks = (1, 2, 5)
ticks = np.outer(major_ticks, minor_ticks).reshape(-1)
ticks = ticks[ticks <= max_epochs]
plot_time_series(time_series=ticks, seed=seednr)

# QUANTITATIVE: sampling, estimating likelihoods and plotting rsq
plot_rsq('gauss', 'ae', quant_runs)
plot_rsq('gauss', 'gae', quant_runs)
plot_rsq('gauss', 'vae', quant_runs)
