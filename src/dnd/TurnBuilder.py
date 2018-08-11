from typing import *
import pmf
from pmf.Pmf import Pmf
from .Attack import Attack
from .AttackBuilder import AttackBuilder
from .HitOutcome import HitOutcome
import itertools
import functools
from .types import *


class TurnBuilder:
    def __init__(self):
        self.attacks = []
        self.turn_damage_roll = 0
        self.turn_damage_bonus = 0

    def attack(self, attack: Union[Attack, AttackBuilder], times: int = 1):
        self.attacks.extend(itertools.repeat(attack if isinstance(attack, Attack) else attack.build(), times))
        return self

    def resolve_turn_extra_damage(self, hit_outcome: HitOutcome) -> Pmf[int]:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return pmf.to_pmf(self.turn_damage_roll).times(2) + self.turn_damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return self.turn_damage_roll + self.turn_damage_bonus
        else:
            return pmf.to_pmf(0)

    def resolve(self, armor_class: IntDist) -> Pmf[int]:
        attack_outcomes = [attack.resolve(armor_class) for attack in self.attacks]
        hardest_hit: Pmf[HitOutcome] = functools.reduce(max, map(lambda outcome: outcome.hit_outcome, attack_outcomes))
        total_damage = sum(map(lambda outcome: outcome.damage, attack_outcomes)) + hardest_hit.map_nested(self.resolve_turn_extra_damage)
        return total_damage

    def dmgroll(self, turn_damage_roll: IntDist):
        self.turn_damage_roll += turn_damage_roll
        return self

    def dmgbon(self, turn_damage_bonus: int):
        self.turn_damage_bonus += turn_damage_bonus
        return self
