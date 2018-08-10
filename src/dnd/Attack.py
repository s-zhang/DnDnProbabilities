from pmf.Pmf import Pmf


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
