from pmf.Pmf import Pmf
from typing import *
from .HitOutcome import HitOutcome
from .AttackOutcome import AttackOutcome
import pmf


class Attack:
    def __init__(self,
                 damage_base: Pmf[int],
                 damage_bonus: int,
                 attack_roll: Pmf[int],
                 attack_bonus: Pmf[int],
                 critical_threshold: int):
        self.damage_base = damage_base
        self.damage_bonus = damage_bonus
        self.attack_roll = attack_roll
        self.attack_bonus = attack_bonus
        self.critical_threshold = critical_threshold

    def resolve_hit(self, armor_class: Pmf[int]) -> Pmf[HitOutcome]:
        def resolve_hit_helper(roll: int) -> Union[HitOutcome, Pmf[HitOutcome]]:
            if roll >= self.critical_threshold:
                return HitOutcome.CRITICAL_HIT
            elif roll == 1:
                return HitOutcome.CRITICAL_MISS
            else:
                return pmf.if_(roll + self.attack_bonus >= armor_class, HitOutcome.NORMAL_HIT, HitOutcome.NORMAL_MISS)
        hit_outcome = self.attack_roll.map_nested(resolve_hit_helper)
        return hit_outcome

    def resolve(self, armor_class: Pmf[int]) -> Pmf[AttackOutcome]:
        def resolve_attack_damage(hit_outcome: HitOutcome) -> Pmf[int]:
            if hit_outcome == HitOutcome.CRITICAL_HIT:
                return self.damage_base + self.damage_base + self.damage_bonus
            elif hit_outcome == HitOutcome.NORMAL_HIT:
                return self.damage_base + self.damage_bonus
            else:
                return pmf.to_pmf(0)

        def resolve_attack_outcome(hit_outcome: HitOutcome) -> Pmf[AttackOutcome]:
            damage_dist = resolve_attack_damage(hit_outcome)
            return damage_dist.map_pmf(lambda damage: AttackOutcome(hit_outcome, damage))

        hit = self.resolve_hit(armor_class)
        return hit.map_nested(resolve_attack_outcome)
