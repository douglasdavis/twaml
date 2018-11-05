import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np


def compare_distributions(dist1, dist2, bins: Optional[np.ndarray] = None,
                          titles: List[str] = ['dist1', 'dist2'],
                          colors: List[str] = ['C0', 'C1'], ratio: bool = True,
                          **subplots_kw):
    """Compare two histogrammed distributons with matplotlib

    Parameters
    ----------
    dist1
      Any mpl-histogrammable object (``np.ndarray``, ``pd.Series``, etc.)
    dist2
      Any mpl-histogrammable object (``np.ndarray``, ``pd.Series``, etc.)
    bins: np.ndarray
      Define the bin edges
    titles: List[str]
      Labels for the distributions
    ratio: bool
      Add a ratio plot
    subplots_kw: Dict
      all additional keywords to send to ``matplotlib.pyplot.subplots``

    Returns
    -------
    fig : matpotlib.figure.Figure

    ax : matplotlib.axes.Axes or array of them
      *ax* can be either a single matplotlib.axes.Axes object or an
      array of Axes objects if more than one subplot was created.  The
      dimensions of the resulting array can be controlled with the
      squeeze keyword, see above.
    h1
      the return of ``matplotlib.axes.Axes.hist`` for dist1
    h2
      the return of ``matplotlib.axes.Axes.hist`` for dist2
    """
    if ratio:
        fig, ax = plt.subplots(2, 1, sharex=True,
                               gridspec_kw=dict(
                                   height_ratios=[3, 1], hspace=.025
                               ), **subplots_kw)
        h1 = ax[0].hist(dist1, bins=bins,  histtype='step', label=titles[0],
                        color=colors[0])
        h2 = ax[0].hist(dist2, bins=h1[1], histtype='step', label=titles[1],
                        color=colors[1])
        centers = np.delete(h1[1], [0])-(np.ediff1d(h1[1])/2.0)
        ax[1].plot(centers, h1[0]/h2[0], 'k-')
        ax[1].plot([centers[0] - 10e3, centers[1] + 10e3], np.ones(2), 'k--')
        ax[1].set_ylim([0, 2])
        ax[1].set_xlim([h1[1][0], h1[1][-1]])
    else:
        fig, ax = plt.subplots(**subplots_kw)
        h1 = ax.hist(dist1, bins=bins,  histtype='step', label=titles[0],
                     color=colors[0])
        h2 = ax.hist(dist2, bins=h1[1], histtype='step', label=titles[1],
                     color=colors[1])
        ax.set_xlim([h1[1][0], h1[1][-1]])

    return fig, ax, h1, h2
