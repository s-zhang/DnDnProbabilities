from typing import *
from pmf.Pmf import Pmf
import matplotlib.pyplot as plt


def plot_at_least(pmf: Pmf[int], name: str = ""):
    outcomes, cdf = pmf.at_least()
    if name == "":
        label = str(pmf.stats())
    else:
        label = "{}: {}".format(pmf.stats(), name)
    plt.scatter(list(outcomes), list(cdf), label=label)


def plot_builds(builds: Dict[str, Pmf[int]]):
    plt.clf()
    ax = plt.subplot(111)

    plt.grid(b=None, which='major', axis='both')
    plt.ylabel('Probability')

    for name, damage in builds.items():
        plot_at_least(damage, name)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=1)

    plt.show()
