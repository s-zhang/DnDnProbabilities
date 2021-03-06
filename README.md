# DnDnProbabilities

Calculates and plot dice roll probabilities for DnD 5e.

## Examples
```python
from dnd import *
from utils import *


AC = 13

"""
barbarian 5, paladin 3, great sword, great weapon fighter,
greate weapon master, rage, reckless attack, sacred weapon,
frenzy, divine smite
"""
builds = {
    "barb 5, paladin 3": turn()
        .attack(attack(d(6).ge(3).times(2)).prof(3).amod(4).adv().gwm().attbon(3).dmgbon(2), times=3)
        .dmgroll(d(2, 8))
        .resolve(AC),
}

plot_builds(builds)
```

![barb paladin build turn damage](/docs/example_plot.png "barb paladin build turn damage")
