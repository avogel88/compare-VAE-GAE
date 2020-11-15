"""
Utility to generate PGF vector files from Python's Matplotlib plots to use in LaTeX documents.
Read more at https://github.com/knly/texfig
"""

import matplotlib as mpl
from math import sqrt


def pgf():
    default_width = 5.78853  # in inches
    default_ratio = (sqrt(5.0) - 1.0) / 2.0  # golden mean
    mpl.use('pgf')
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "figure.figsize": [default_width, default_width * default_ratio],
        "pgf.preamble": [
            # put LaTeX preamble declarations here
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            # macros defined here will be available in plots, e.g.:
            r"\newcommand{\vect}[1]{#1}",
            # You can use dummy implementations, since your LaTeX document
            # will render these properly, anyway.
        ],
    })
