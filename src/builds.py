from dnd import *
from utils import *


AC = 16

"""
pam: polearm master
h's mark: hunter's mark
gwm: great weapon master
gwf: great weapon fighter
cbe: crossbow expert
ss: sharpshooter
"""
builds = {
    "paladin 5, barb 2, hexblade 1, pam, gwm": turn()
    .attack(attack(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(3 + 2).crit(19), times=2)
    .attack(attack(d(4)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(3 + 2).crit(19))
    .resolve(AC),
#    "hexblade 8, elven accuracy, gwm": turn()
#    .attack(attack(d(2, 6)).prof(3).amod(4).adv(2).gwm().dmgbon(3 + 1).crit(19), times=2)
#    .resolve(AC),
#    "Guiding bolt x2 + spiritual weapon": turn()
#    .attack(attack(d(4, 6)).prof(3).amod(4, False).attbon(d(1, 4)), times=2)
#    .attack(attack(d(2, 8)).prof(3).amod(4).attbon(d(1, 4)))
#    .resolve(AC),
#    "Inflict wounds x2 + spiritual weapon": turn()
#    .attack(attack(d(3, 10)).prof(3).amod(4, False).attbon(d(1, 4)), times=2)
#    .attack(attack(d(2, 8)).prof(3).amod(4).attbon(d(1, 4)))
#    .resolve(AC),
    "hexblade 5, fighter 1, elven accuracy, ss": turn()
    .attack(attack(d(1, 10)).prof(3).amod(4).attbon(2 + 1).dmgbon(3 + 1).adv(2).gwm().crit(19), times=2)
    .resolve(AC),
#    "ranger 5, pam, quarterstaff, h's mark, shield": turn()
    # can get 1 lvl of druid or nature cleric to get shillelagh to change
    # base damage to 1d8, but damage increase is minimal. Better stack dex
    # to avoid losing conc. of h's mark.
#    .attack(attack(d(6) + d(6)).prof(3).amod(5).dmgbon(2), times=3)
#    .attack(attack(d(4) + d(6)).prof(3).amod(5).dmgbon(2))
#    .resolve(AC),
#    "monk 5, warlock 1, pam, hex": turn()
#    .attack(attack(d(8) + d(6)).prof(3).amod(5), times=3)
#    .attack(attack(d(6) + d(6)).prof(3).amod(5), times=2)
#    .resolve(AC),
    "barb 5, paladin 3, GWF, GWM, frenzy": turn()
    .attack(attack(d(6).ge(3).times(2)).prof(3).amod(4).adv().gwm().attbon(3).dmgbon(2), times=3)
    .resolve(AC),
    "v paladin 5, hexblade 1, barb 1": turn()
    .attack(attack(d(1, 8)).prof(3).amod(4).dmgbon(3 + 2).adv(2).crit(19), times=2)
    .attack(attack(d(1, 6)).prof(3).amod(4, False).dmgbon(3 + 2).adv(2).crit(19))
    .dmgbon(d(2, 8))
    .resolve(AC),
    "hexblade 5, rogue 3, elven accuracy": turn()
    .attack(attack(d(1, 8)).prof(3).amod(4).attbon(1).dmgbon(3 + 1).adv(2).crit(19), times=2)
    .attack(attack(d(1, 6)).prof(3).amod(4, False).attbon(1).dmgbon(3 + 1).adv(2).crit(19))
    .dmgbon(d(2, 6))
    .resolve(AC),
    "hexblade 5, paladin 4, elven accuracy, GWM": turn()
    .attack(attack(d(2, 8)).prof(4).amod(3).attbon(1).dmgbon(3 + 1).adv(2).gwm().crit(19), times=2)
    .resolve(AC),
#    "Guiding bolt x2 + hex": turn()
#    Very weak damage
#    .attack(attack(d(5, 6)).prof(3).amod(4, False).adv(), times=2)
#    .resolve(AC)
#    "paladin 2, rogue 3, gwm, crit": turn()
#    .attack(attack(d(6).ge(3).times(2)).prof(3).amod(1).gwm().dmgroll(d(2, 8)).crit(0), times=2)
#    .resolve(AC),
    #"berserker 5, hexblade 1, gwm, crit, half-orc": turn()
    #.attack(attack(d(1, 12)))
#    "vengeance paladin 5, hexblade 1"
}


def make_sample_turn(ac):
    sample_turn = turn()\
        .attack(attack(d(1)).prof(3), times=1)\
        .resolve(ac)
    return sample_turn


#builds = {
#    "AC: " + str(ac): make_sample_turn(ac) for ac in range(10, 20)
#}
#print(make_sample_turn(17).stats())
plot_builds(builds)
from pmf import *
#print(d(10, 4).stats())
#print(ints([0, 0.5, 0.25, 0.25], 1).times(10).stats())
#print(ints([1/8/2, 1/8/2, 1/8/2, 1/8/2, 1/8*1.5, 1/8*1.5, 1/8*1.5, 1/8*1.5], 1).stats())
#print(ints([1/6/2, 1/6/2, 1/6/2, 1/6*1.5, 1/6*1.5, 1/6*1.5], 1).stats())
