from typing import *
import matplotlib.pyplot as plt
from pmf.Pmf import Pmf
from dnd import *


test_damage1 = resolve_turn_attacks(AttackBuilder(d(10)).prof(3).amod(3).gwm().dmgroll(d(3, 8)).crit(0).times(2),
                                    AttackBuilder(d(4)).prof(3).amod(3).gwm().dmgroll(d(2, 8)).crit(0),
                                    armor_class=15)

print(test_damage1.stats())

AC = 13

# pam: polearm master
# h's mark: hunter's mark
builds = {
    "paladin 5, barb 2, pam, gwm":
        resolve_turn_attacks(AttackBuilder(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2).times(2),
                             AttackBuilder(d(4)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2),
                             extra_damage_roll=d(2, 8),
                             armor_class=AC),
    "ranger 5, rogue 3, crossbow expert, sharpshooter":
        resolve_turn_attacks(AttackBuilder(d(2, 6)).prof(3).amod(4).attbon(2).gwm().times(3),
                             extra_damage_roll=d(8) + d(2, 6),
                             armor_class=AC),
    "ranger 5, pam, quarterstaff, h's mark, shield":
        # can get 1 lvl of druid or nature cleric to get shillelagh to change base damage to 1d8, but
        # damage increase is minimal. Better stack dex to avoid losing conc. of h's mark.
        resolve_turn_attacks(AttackBuilder(d(6) + d(6)).prof(3).amod(5).dmgbon(2).times(3),
                             AttackBuilder(d(4) + d(6)).prof(3).amod(5).dmgbon(2),
                             armor_class=AC),
    "monk 5, warlock 1, pam, hex":
        resolve_turn_attacks(AttackBuilder(d(8) + d(6)).prof(3).amod(5).times(3),
                             AttackBuilder(d(6) + d(6)).prof(3).amod(5).times(2),
                             armor_class=AC),
    "monk 5, ranger 3, warlock 1, pam, hex":
        resolve_turn_attacks(AttackBuilder(d(8) + d(6)).prof(4).amod(5).dmgbon(2).times(3),
                             AttackBuilder(d(6) + d(6)).prof(4).amod(5).dmgbon(2).times(2),
                             armor_class=AC),
    "barb 5, paladin 3, GWM, frenzy":
        resolve_turn_attacks(AttackBuilder(d(6).ge(3).times(2)).prof(3).amod(4).adv().gwm().attbon(3).dmgbon(2).times(3),
                             armor_class=AC)
}


do_plot = True
if do_plot:
    def plot_stats(pmf: Pmf[int], name: str = ""):
        outcomes, cdf = pmf.at_least()
        if name == "":
            label = str(pmf.stats())
        else:
            label = "{}: {}".format(pmf.stats(), name)
        plt.scatter(list(outcomes), list(cdf), label=label)

    plt.clf()
    ax = plt.subplot(111)

    plt.grid(b=None, which='major', axis='both')
    plt.ylabel('Probability')

    for name, damage in builds.items():
        plot_stats(damage, name)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=1)

    plt.show()
