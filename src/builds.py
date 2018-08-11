from dnd import *
from utils import *


AC = 13

"""
pam: polearm master
h's mark: hunter's mark
gwm: great weapon master
gwf: great weapon fighter
"""
builds = {
    "paladin 5, barb 2, pam, gwm": turn()
    .attack(attack(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2), times=2)
    .attack(attack(d(4)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2))
    .dmgroll(d(2, 8))
    .resolve(AC),
    "ranger 5, rogue 3, crossbow expert, sharpshooter": turn()
    .attack(attack(d(2, 6)).prof(3).amod(4).attbon(2).gwm(), times=3)
    .dmgroll(d(8) + d(2, 6))
    .resolve(AC),
    "ranger 5, pam, quarterstaff, h's mark, shield": turn()
    # can get 1 lvl of druid or nature cleric to get shillelagh to change
    # base damage to 1d8, but damage increase is minimal. Better stack dex
    # to avoid losing conc. of h's mark.
    .attack(attack(d(6) + d(6)).prof(3).amod(5).dmgbon(2), times=3)
    .attack(attack(d(4) + d(6)).prof(3).amod(5).dmgbon(2))
    .resolve(AC),
    "monk 5, warlock 1, pam, hex": turn()
    .attack(attack(d(8) + d(6)).prof(3).amod(5), times=3)
    .attack(attack(d(6) + d(6)).prof(3).amod(5), times=2)
    .resolve(AC),
    "monk 5, ranger 3, warlock 1, pam, hex": turn()
    .attack(attack(d(8) + d(6)).prof(4).amod(5).dmgbon(2), times=3)
    .attack(attack(d(6) + d(6)).prof(4).amod(5).dmgbon(2), times=2)
    .resolve(AC),
    "barb 5, paladin 3, GWF, GWM, frenzy": turn()
    .attack(attack(d(6).ge(3).times(2)).prof(3).amod(4).adv().gwm().attbon(3).dmgbon(2), times=3)
    .resolve(AC)
#    "paladin 5, rogue 3, pam, gwm, crit": turn()
#        .attack(attack(d(10)).prof(3).amod(3).gwm().dmgroll(d(3, 8)).crit(0), times=2)
#        .attack(attack(d(4)).prof(3).amod(3).gwm().dmgroll(d(2, 8)).crit(0))
#        .resolve(AC)
}

plot_builds(builds)
