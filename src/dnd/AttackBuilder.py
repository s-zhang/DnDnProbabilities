from .types import IntDist
from pmf.Pmf import Pmf
from .dice import d
from .Attack import Attack
import pmf


class AttackBuilder:
    def __init__(self, damage_base: IntDist):
        self.damage_base = damage_base
        self.damage_bonus = 0
        self.n_adv = 0
        self.is_lucky = False
        self.proficiency_bonus = 0
        self.ability_modifier = 0
        self.add_ability_modifier_to_damage = True
        self.attack_bonus: IntDist = 0
        self.critical_threshold = 20
        self.is_great_weapon_master_or_sharpshooter = False

    def dmgroll(self, extra_damage_roll: IntDist):
        self.damage_base += extra_damage_roll
        return self

    def dmgbon(self, damage_bonus: int):
        self.damage_bonus += damage_bonus
        return self

    def adv(self, n=1):
        self.n_adv = n
        return self

    def lucky(self, is_lucky=True):
        self.is_lucky = is_lucky
        return self

    def prof(self, proficiency_bonus: int):
        self.proficiency_bonus = proficiency_bonus
        return self

    def amod(self, ability_modifier: int, add_ability_modifier_to_damage=True):
        self.ability_modifier = ability_modifier
        self.add_ability_modifier_to_damage = add_ability_modifier_to_damage
        return self

    def attbon(self, attack_bonus: IntDist):
        self.attack_bonus += attack_bonus
        return self

    def crit(self, critical_threshold: int):
        self.critical_threshold = critical_threshold
        return self

    def gwm(self, is_great_weapon_master_or_sharpshooter=True):
        self.is_great_weapon_master_or_sharpshooter = \
            is_great_weapon_master_or_sharpshooter
        return self

    @staticmethod
    def __lucky_roll(attack_roll: Pmf[int]):
        return attack_roll.map_nested(
            lambda roll: attack_roll if roll == 1 else roll)

    def build(self):
        damage_bonus = self.damage_bonus
        if self.add_ability_modifier_to_damage:
            damage_bonus += self.ability_modifier

        attack_roll = d(20)

        if self.is_lucky:
            if self.n_adv != 0:
                raise NotImplementedError
            else:
                attack_roll = self.__lucky_roll(attack_roll)
        else:
            attack_roll = attack_roll.adv(self.n_adv)

        attack_bonus = self.attack_bonus + \
            self.proficiency_bonus + \
            self.ability_modifier

        if self.is_great_weapon_master_or_sharpshooter:
            damage_bonus += 10
            attack_bonus -= 5

        return Attack(pmf.to_pmf(self.damage_base),
                      damage_bonus,
                      attack_roll,
                      pmf.to_pmf(attack_bonus),
                      self.critical_threshold)

    def resolve(self, armor_class: IntDist) -> Pmf[int]:
        attack = self.build()
        return attack.resolve(pmf.to_pmf(armor_class)).map_pmf(
            lambda outcome: outcome.damage)
