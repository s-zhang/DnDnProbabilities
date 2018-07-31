import lea
from typing import Dict, Union, Callable, Any
import operator
from collections import OrderedDict
from lea import Lea
from enum import Enum


Distribution = Lea
Dist = Union[Distribution, int]


def multiple_dice(number_of_dice: int, number_of_face: int) -> Distribution:
    single_die = die(number_of_face)
    multiple_dice_ = single_die.times(number_of_dice, operator.__add__)
    return multiple_dice_


def die(number_of_face: int) -> Distribution:
    face_probability = 1 / number_of_face
    die_ = lea.pmf({i + 1: face_probability for i in range(number_of_face)})
    return die_


def d(*args) -> Distribution:
    if len(args) == 1:
        return die(args[0])
    elif len(args) == 2:
        return multiple_dice(*args)
    else:
        raise ValueError(args)


def adv(pmf: Distribution) -> Distribution:
    return pmf.times(2, max)


def largest_n_out_of(pmf: Lea, n: int, out_of: int) -> Distribution:
    return pmf.map(lambda outcome: (outcome,))\
        .times(out_of, lambda outcomes1, outcomes2: tuple(sorted(outcomes1 + outcomes2)[-n:]))


def nth_largest_out_of(pmf: Distribution, n: int, out_of: int) -> Distribution:
    return largest_n_out_of(pmf, n, out_of).map(lambda outcomes: outcomes[0])


def at_least(pmf: Distribution) -> Dict:
    at_least_dict = OrderedDict()
    cumulative_p = 1
    for outcome, p in pmf.pmf_dict.items():
        at_least_dict[outcome] = cumulative_p
        cumulative_p -= p
    return at_least_dict


class PmfStats:
    def __init__(self, mean: float, std: float, percentile80: int, percentile20: int):
        self.mean = mean
        self.std = std
        self.percentile80 = percentile80
        self.percentile20 = percentile20

    def __str__(self):
        return "{:.2f} +/- {:.2f}, 80/20p: {}/{}".format(self.mean, self.std, self.percentile80, self.percentile20)


def stats(pmf: Distribution) -> PmfStats:
    percentile80, percentile20 = None, None
    cumulative_p = 1
    for outcome, p in pmf.pmf_dict.items():
        cumulative_p -= p
        if cumulative_p <= 0.8 and percentile80 is None:
            percentile80 = outcome
        if cumulative_p <= 0.2 and percentile20 is None:
            percentile20 = outcome
    return PmfStats(pmf.mean, pmf.std, percentile80, percentile20)


def map_nested(pmf: Distribution, f: Callable[[Any], Dist]) -> Distribution:
    return pmf.map(lambda outcome: lea.coerce(f(outcome))).get_alea().flat()


stat_roll = largest_n_out_of(d(6), 3, 4).map(sum)

assert 0.016203703703703706 == stat_roll.p(18)

second_best_roll = nth_largest_out_of(stat_roll, 2, 6)

assert 0.003771297899701062 == second_best_roll.p(18)

best_roll = nth_largest_out_of(stat_roll, 1, 6)


class Attack:
    def __init__(self,
                 damage_base: Distribution,
                 damage_bonus: int,
                 attack_roll: Distribution,
                 attack_bonus: Distribution,
                 critical_threshold):
        self.damage_base = damage_base
        self.damage_bonus = damage_bonus
        self.attack_roll = attack_roll
        self.attack_bonus = attack_bonus
        self.critical_threshold = critical_threshold


class HitOutcome(Enum):
    CRITICAL_MISS = 0
    NORMAL_MISS = 1
    NORMAL_HIT = 2
    CRITICAL_HIT = 3


class AttackOutcome:
    def __init__(self, hit_outcome: HitOutcome, damage: int):
        self.hit_outcome = hit_outcome
        self.damage = damage

    def __eq__(self, other):
        return self.damage == other.damage and self.hit_outcome == other.hit_outcome

    def __hash__(self):
        return hash((self.damage, self.hit_outcome))

    def __str__(self):
        return "[" + str(self.damage) + " from " + str(self.hit_outcome) + "]"


def resolve_hit(attack: Attack, armor_class: Distribution) -> Distribution:
    def resolve_hit_helper(roll: int) -> Union[HitOutcome, Distribution]:
        if roll >= attack.critical_threshold:
            return HitOutcome.CRITICAL_HIT
        elif roll == 1:
            return HitOutcome.CRITICAL_MISS
        else:
            return lea.if_(roll + attack.attack_bonus >= armor_class, HitOutcome.NORMAL_HIT, HitOutcome.NORMAL_MISS)
    return map_nested(attack.attack_roll, resolve_hit_helper)


def resolve_attack(attack: Attack, armor_class: Distribution) -> Distribution:
    def resolve_attack_damage(hit_outcome: HitOutcome) -> Distribution:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return attack.damage_base.times(2, operator.__add__) + attack.damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return attack.damage_base + attack.damage_bonus
        else:
            return lea.coerce(0)

    def resolve_attack_outcome(hit_outcome: HitOutcome) -> Distribution:
        damage_dist = resolve_attack_damage(hit_outcome)
        return damage_dist.map(lambda damage: AttackOutcome(hit_outcome, damage))

    hit = resolve_hit(attack, armor_class)
    return hit.map(resolve_attack_outcome).get_alea().flat()


class AttackBuilder:
    def __init__(self, damage_base: Dist):
        self.damage_base = damage_base
        self.damage_bonus = 0
        self.is_adv = False
        self.proficiency_bonus = 0
        self.ability_modifier = 0
        self.add_ability_modifier_to_damage = True
        self.attack_bonus: Dist = 0
        self.critical_threshold = 20
        self.is_great_weapon_master_or_sharpshooter = False

    def dmgroll(self, extra_damage_roll: Dist):
        self.damage_base += extra_damage_roll
        return self

    def dmgbon(self, bonus_damage: int):
        self.damage_bonus += bonus_damage
        return self

    def adv(self):
        self.is_adv = True
        return self

    def prof(self, proficiency_bonus: int):
        self.proficiency_bonus = proficiency_bonus
        return self

    def amod(self, ability_modifier: int, add_ability_modifier_to_damage=True):
        self.ability_modifier = ability_modifier
        self.add_ability_modifier_to_damage = add_ability_modifier_to_damage
        return self

    def attbon(self, attack_bonus: Dist):
        self.attack_bonus += attack_bonus
        return self

    def crit(self, critical_threshold: int):
        self.critical_threshold = critical_threshold
        return self

    def gwm(self):
        self.is_great_weapon_master_or_sharpshooter = True
        return self

    def build(self):
        bonus_damage = self.damage_bonus
        if self.add_ability_modifier_to_damage:
            bonus_damage += self.ability_modifier

        attack_roll = d(20)
        if self.is_adv:
            attack_roll = adv(attack_roll)

        attack_bonus = self.attack_bonus + self.proficiency_bonus + self.ability_modifier

        if self.is_great_weapon_master_or_sharpshooter:
            bonus_damage += 10
            attack_bonus -= 5

        return Attack(self.damage_base, bonus_damage, attack_roll, attack_bonus, self.critical_threshold)

    def resolve(self, armor_class: Dist) -> Distribution:
        attack = self.build()
        return resolve_attack(attack, lea.coerce(armor_class)).damage


assert 0.0486111111111111 == AttackBuilder(d(6)).resolve(15).p(6)

assert 1 == AttackBuilder(d(1)).attbon(-20).crit(0).resolve(0).p(2)

AC = 15
test_damage = AttackBuilder(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2).resolve(AC).times(2) + \
              AttackBuilder(d(4)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2).resolve(AC)

assert 45.19125000000002 == test_damage.mean

print(stats(test_damage))

"""
import matplotlib.pyplot as plot
cdf = at_least(test_damage)


plot.ion()
plot.clf()
plot.bar(cdf.keys(), cdf.values(), align='center')
plot.ylabel('Probability')
plot.show()
"""