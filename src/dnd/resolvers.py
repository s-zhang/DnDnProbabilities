from typing import *
from .HitOutcome import HitOutcome
from .Attack import Attack
from .AttackOutcome import AttackOutcome
import pmf
from pmf.Pmf import Pmf
from .types import IntDist
import itertools
import functools


def resolve_hit(attack: Attack, armor_class: Pmf[int]) -> Pmf[HitOutcome]:
    def resolve_hit_helper(roll: int) -> Union[HitOutcome, Pmf[HitOutcome]]:
        if roll >= attack.critical_threshold:
            return HitOutcome.CRITICAL_HIT
        elif roll == 1:
            return HitOutcome.CRITICAL_MISS
        else:
            return pmf.if_(roll + attack.attack_bonus >= armor_class, HitOutcome.NORMAL_HIT, HitOutcome.NORMAL_MISS)
    hit_outcome = attack.attack_roll.map_nested(resolve_hit_helper)
    return hit_outcome


def resolve_attack(attack: Attack, armor_class: Pmf[int]) -> Pmf[AttackOutcome]:
    def resolve_attack_damage(hit_outcome: HitOutcome) -> Pmf[int]:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return attack.damage_base + attack.damage_base + attack.damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return attack.damage_base + attack.damage_bonus
        else:
            return pmf.to_pmf(0)

    def resolve_attack_outcome(hit_outcome: HitOutcome) -> Pmf[AttackOutcome]:
        damage_dist = resolve_attack_damage(hit_outcome)
        return damage_dist.map_pmf(lambda damage: AttackOutcome(hit_outcome, damage))

    hit = resolve_hit(attack, armor_class)
    return hit.map_nested(resolve_attack_outcome)


def resolve_turn_attacks(*attacks,
                         armor_class: IntDist,
                         extra_damage_roll: IntDist = 0,
                         damage_bonus: int = 0) -> Pmf[int]:
    def resolve_turn_extra_damage(hit_outcome: HitOutcome) -> Pmf[int]:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return pmf.to_pmf(extra_damage_roll).times(2) + damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return extra_damage_roll + damage_bonus
        else:
            return pmf.to_pmf(0)
    attacks = itertools.chain.from_iterable(map(lambda attack: attack if isinstance(attack, List) else [attack], attacks))
    attacks = map(lambda attack: attack if isinstance(attack, Attack) else attack.build(), attacks)
    attacks = list(map(lambda attack: resolve_attack(attack, armor_class), attacks))
    hardest_hit: Pmf[HitOutcome] = functools.reduce(max, map(lambda attack: attack.hit_outcome, attacks))
    total_damage = sum(map(lambda attack: attack.damage, attacks)) + hardest_hit.map_nested(resolve_turn_extra_damage)
    return total_damage
