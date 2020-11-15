import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname
import pandas as pd
from scipy.stats import norm

from .experiment import config
from ..tensorboard import get_losses, tabulate_tensors
from ..tensorflow.keras.datasets import import_mnist

__all__ = ['hor', 'plot_alt_dist', 'plot_history', 'plot_ks', 'plot_marginal_gauss', 'plot_metrics', 'plot_mnist_dist',
           'plot_rsq', 'plot_strip', 'strip', 'vert']


def vert(x):
    """Stack images vertically."""
    return x.reshape(np.prod(x.shape[:2]), -1)


def hor(x):
    """Stack images horizontally."""
    return np.concatenate(x, axis=1)


def strip(x, *y):
    """Stack images horizontally and different lists (y) vertically."""
    if y != ():
        return np.vstack([strip(x)] + [strip(z) for z in y])
    return np.expand_dims(np.concatenate(x, axis=1), 0)


def plot_mnist_dist():
    """Determine and plot distributions of MNIST and Fashion-MNIST."""
    sign = -1
    width = .45
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True, sharey=True, gridspec_kw={'wspace': 0})
    (_, x), (_, y) = import_mnist()
    for z, label in ((x, 'train'), (y, 'test')):
        h, p = np.unique(z, return_counts=True)
        p = p / sum(p)
        pos = sign * width / 2
        sign *= -1
        axes[0].bar(h + pos, p, width, label=label)  # , color='white', edgecolor='black', hatch='/')
    (_, x), (_, y) = import_mnist(variant='fashion')
    for z, label in ((x, 'train'), (y, 'test')):
        h, p = np.unique(z, return_counts=True)
        p = p / sum(p)
        pos = sign * width / 2
        sign *= -1
        axes[1].bar(h + pos, p, width, label=label)
    axes[0].set_title('MNIST')
    axes[1].set_title('Fashion-MNIST')

    for ax in axes:
        ax.set(xlabel='Moden', ylabel='p')
        ax.set_ylim(bottom=.08)
        ax.set_yticks(np.linspace(.08, .12, 5))
        ax.set_xticks(range(10))
        ax.label_outer()
        ax.legend(frameon=False, loc=1)

    plt.tight_layout()
    file = config('PATHS')['mnist_pdfs']
    os.makedirs(dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.close()


def plot_alt_dist():
    """Plot alternate distributions for further work."""
    fig = plt.figure(figsize=(3, 2))
    x = np.linspace(-3, 3, 100)
    y = np.linspace(0, 9, 100)
    plt.plot(y, norm.pdf(x, scale=1))
    plt.plot(y, norm.pdf(x, scale=2))
    plt.xticks(range(0, 10, 2))
    plt.xlabel('Moden')
    plt.ylabel('p')
    plt.legend(['train', 'test'], frameon=False, loc=1)
    plt.tight_layout()
    file = config('PATHS')['alt_pdfs']
    os.makedirs(dirname(file), exist_ok=True)
    plt.savefig(file)
    plt.close()


def plot_strip(images,
               n: int = 20,
               xlabels=None,
               ylabels=None,
               headers=None,
               path: str = ''):
    """
    Plot strips of images.
    shape: rows, columns, dimy, dimx, channels
    """
    images = [i[:n] for i in images]
    dim = np.ndim(images)
    images = np.expand_dims(images, axis=list(range(5 - dim)))
    rows, columns, dimy, dimx, channels = np.shape(images)
    fig, axes = plt.subplots(rows, columns, figsize=(6, 2),
                             gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    for row in range(rows):
        for col in range(columns):
            ax = axes
            if rows > 1:
                ax = axes[row]
            ax[col].imshow(images[row, col], cmap=plt.cm.gray, vmin=0, vmax=1)
            # ax[col].axis('off')
            if headers is not None:
                ax[col].set_title(headers[col])
            if xlabels is not None:
                ax[col].set_xlabel(xlabels[col])
            if ylabels is not None:
                ax[col].set_ylabel(ylabels[row])  # ,rotation=0, labelpad=20
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()
    for ax in axes[-1]:
        ax.set_xticks([0, dimx])
        ax.set_xticklabels([])
    if path:
        os.makedirs(dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_metrics(x, val, title, ax):
    if type(x) == int:
        x = range(1, x + 1)

    mean = val.mean(axis=0)
    std = val.std(axis=0)

    ax.loglog(x, mean, 'k')
    ax.loglog(x, mean + std, 'k-.', linewidth=.8)
    ax.loglog(x, mean - std, 'k-.', linewidth=.8)
    ax.set_xlabel('Epochen')
    ax.set_title(title)


def plot_ks(dataset, runs, epoch):
    """Read ks statistics and generate statistics over runs."""
    ks = []
    for run in range(runs):
        # load run wise ks statistics
        ex = {
            'dataset': dataset,
            'model'  : 'gae',
            'run'    : run,
            'epoch'  : epoch
        }
        ks += [np.load(config('PATHS', **ex)['kstest'].format(epoch=epoch))]
    # intersection of ks statistics
    l = min([len(sub_ks) for sub_ks in ks])
    ks = np.array([sub_ks[:l] for sub_ks in ks])

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    plot_metrics(l, ks,
                 'Kolmogorow-Smirnow-Test',
                 ax)
    path = config('PATHS', **ex)['kse_img']
    os.makedirs(dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close()


def plot_history(
        dataset,
        model,
        runs,
        maxepoch: int = 10000):
    """Plot metrics of DNN."""
    num_losses = 2
    hist = np.zeros((runs, num_losses, maxepoch))
    for run in range(runs):
        hist[run] = get_losses(config(
            'PATHS',
            dataset=dataset,
            model=model,
            run=run)['root']).to_numpy()[:maxepoch].T

    fig, axes = plt.subplots(1, num_losses, figsize=(6, 2),
                             sharey=True,
                             gridspec_kw={'hspace': 0, 'wspace': 0})
    titles = ['Verlust', 'Validierungsverlust']

    for i in range(num_losses):
        plot_metrics(maxepoch, hist[:, i], titles[i], axes[i])
    path = config('PATHS', dataset=dataset, model=model)['hist_img']
    os.makedirs(dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close()


def plot_rsq(dataset, model, runs):
    corr = []
    cols = ['rec_corr', 'enc_corr', 'gen_corr']
    for run in range(runs):
        wall_time, steps, data = tabulate_tensors(
            config(
                'PATHS',
                dataset=dataset,
                model=model,
                run=run)['rsq'])
        df = pd.DataFrame(data)[cols]
        corr += [df.to_numpy(dtype=np.float32)]
    corr = np.array(corr)
    avg = corr.mean(axis=0).T
    std = corr.std(axis=0).T
    fig, axes = plt.subplots(
        1, 3, figsize=(6, 2),
        sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0})
    title = ['Rekonstruktionen', 'Encodings', 'generierte Beispiele']
    x = steps
    for i, col in enumerate(cols):
        axes[i].plot(x, avg[i], 'k')
        axes[i].plot(x, avg[i] + std[i], 'k-.', linewidth=.8)
        axes[i].plot(x, avg[i] - std[i], 'k-.', linewidth=.8)
        axes[i].set_title(title[i])
    for ax in axes.flat:
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel('r²')
        ax.set_xlabel('Epochen')
        ax.ticklabel_format(
            axis='x', style='sci',
            scilimits=(0, 0), useMathText=True)
        ax.label_outer()
        ax.set_xlim((0, steps[-1] - 1))
    # save mean +/- std of correlations
    corr = np.empty((2*avg.shape[0], avg.shape[1]), dtype=avg.dtype)
    corr[0::2] = avg
    corr[1::2] = std

    ex = {
        'dataset': dataset,
        'model'  : model,
        'run'    : runs
    }
    os.makedirs(dirname(config('PATHS', **ex)['rsq_img']), exist_ok=True)
    np.savetxt(config('PATHS', **ex)['rsq_avg'], corr)
    fig.savefig(config('PATHS', **ex)['rsq_img'], bbox_inches='tight')
    plt.close()


def plot_marginal_gauss(x, stats, eppf, file=None):
    """Schematic for the marginal gaussianization."""
    fig, axes = plt.subplots(
        2, 2, figsize=(6, 4),
        gridspec_kw={'hspace': 0, 'wspace': 0})

    # prepare axes
    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.tick_right()
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()
    axes[0, 0].set_xticks([])
    axes[0, 1].set_xticks([])
    axes[1, 0].set_xlabel(r'empirische Beispiele $x_1$')
    axes[1, 1].set_xlabel(r'normalverteilte Beispiele $x_2$')
    axes[0, 0].set_ylabel('PDF')
    axes[0, 1].set_ylabel('PDF')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 1].set_ylabel('CDF')

    # plots
    axes[0, 0].plot(x[0], stats[0].pdf(x[0]), c='black')
    axes[0, 1].plot(x[1], stats[1].pdf(x[1]), c='black')
    axes[1, 0].plot(x[0], stats[0].cdf(x[0]), c='black')
    axes[1, 1].plot(x[1], stats[1].cdf(x[1]), c='black')

    # calculate values
    ecdf = stats[0].cdf(eppf)
    nppf = stats[1].ppf(ecdf)
    epdf = stats[0].pdf(eppf)
    npdf = stats[1].pdf(ecdf)

    def limits(ax):
        """Getting & setting axes limits."""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        return (xmin, xmax), (ymin, ymax)

    # get limits
    (epxmin, epxmax), (epymin, epymax) = limits(axes[0, 0])
    (npxmin, npxmax), (npymin, npymax) = limits(axes[0, 1])
    # linking y axes of CDFs
    axes[1, 0].set_ylim(axes[1, 1].get_ylim())
    (ecxmin, ecxmax), (ecymin, ecymax) = limits(axes[1, 0])
    (ncxmin, ncxmax), (ncymin, ncymax) = limits(axes[1, 1])

    black = {
        'colors': 'k',
        'linestyles': 'dashdot',
        'linewidth': .8,
    }
    red = {
        'colors': 'r',
        'linestyles': 'dashdot',
        'linewidth': .8,
    }

    # draw example
    axes[0, 0].hlines(epdf, epxmin, eppf, **black)
    axes[0, 0].vlines(eppf, epymin, epdf, **black)
    axes[1, 0].vlines(eppf, ecdf, ecymax, **black)
    axes[1, 0].vlines(eppf, ecymin, ecdf, **red)
    axes[1, 0].hlines(ecdf, eppf, ecxmax, **black)
    axes[1, 1].hlines(ecdf, ncxmin, nppf, **black)
    axes[1, 1].vlines(nppf, ecdf, ncymax, **black)
    axes[1, 1].vlines(nppf, ncymin, ecdf, **red)
    axes[0, 1].vlines(nppf, npymin, npdf, **black)
    axes[0, 1].hlines(npdf, nppf, npxmax, **black)

    # small gaps between lines and text
    def gaps(xmin, xmax, ymin, ymax, percent):
        return (xmax - xmin) / 100 * percent, (ymax - ymin) / 100 * percent

    epxdelta, epydelta = gaps(epxmin, epxmax, epymin, epymax, 2)
    ecxdelta, ecydelta = gaps(ecxmin, ecxmax, ecymin, ecymax, 2)
    npxdelta, npydelta = gaps(npxmin, npxmax, npymin, npymax, 2)
    ncxdelta, ncydelta = gaps(ncxmin, ncxmax, ncymin, ncymax, 2)

    hor = {
        'fontsize': 12,
        'horizontalalignment': 'center',
        'verticalalignment': 'bottom',
    }
    vert = {
        'fontsize': 12,
        'horizontalalignment': 'left',
        'verticalalignment': 'center',
    }

    # annotate values on lines
    axes[0, 0].text(
        (epxmax - epxmin) / 8 + epxmin,
        epdf + epydelta,
        '%.2g' % epdf,
        **hor)
    axes[1, 0].text(
        eppf + ecxdelta,
        (ecymax + ecymin) / 8,
        '%.2g' % eppf,
        **vert)
    axes[1, 0].text(
        (ecxmax - ecxmin) * 7 / 8 + ecxmin,
        ecdf + ecydelta, '%.2g' % ecdf,
        **hor)
    axes[1, 1].text(
        nppf + ncxdelta,
        (ncymax + ncymin) / 8,
        '%.2g' % nppf,
        **vert)
    axes[0, 1].text(
        (npxmax - npxmin) * 7 / 8 + npxmin,
        npdf + npydelta,
        '%.2g' % npdf,
        **hor)

    # scientific y axes
    for ax in axes.flat:
        ax.ticklabel_format(
            axis='y', style='sci',
            scilimits=(0, 0), useMathText=True)

    if file is not None:
        os.makedirs(dirname(file), exist_ok=True)
        fig.savefig(file)
    plt.close()
