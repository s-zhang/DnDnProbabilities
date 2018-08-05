from enum import Enum
from typing import *
import functools
import itertools
import operator
import math
import bisect
import matplotlib.pyplot as plt

TOutcome = TypeVar("TOutcome")
AnyPmf = Union[Any, "Pmf[Any]"]


class Pmf(Generic[TOutcome]):
    def p(self, outcome: TOutcome) -> float:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[TOutcome, float]]:
        raise NotImplementedError

    def outcome_to_str(self, outcome: TOutcome) -> str:
        return str(outcome)

    def __str__(self) -> str:
        lines = []
        for outcome, probability in self:
            lines.append("{}: {:.3f}".format(self.outcome_to_str(outcome), probability))
        return "\n".join(lines)

    @staticmethod
    def coerce(obj: Any) -> "Pmf[Any]":
        if isinstance(obj, Pmf):
            return obj
        elif isinstance(obj, int):
            return IntegerInterval([1], obj)
        else:
            return Constant(obj)

    def map(self, f: Callable[[TOutcome, float], Any]):
        return map(lambda outcome_probability: f(*outcome_probability), self)

    def map_pmf(self, f: Callable[[TOutcome], Any]) -> "Pmf[Any]":
        pmf = {}
        for outcome, probability in self.map(lambda o, p: (f(o), p)):
            pmf[outcome] = pmf.get(outcome, 0) + probability
        return pmf_from_table(pmf)

    def map_nested(self, f: Callable[[TOutcome], Any]) -> "Pmf[Any]":
        return functools.reduce(self.__class__.union, self.map(lambda outcome, probability:
                                                               self.coerce(f(outcome)).scale_probability(probability)))

    def __eq__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__eq__, other)

    def __ge__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__ge__, other)

    def __le__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__le__, other)

    def __gt__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__gt__, other)

    def __lt__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__lt__, other)

    def __or__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__or__, other)

    def __ror__(self, other) -> "Pmf[bool]":
        return self.bool_op(operator.__or__, other)

    def __add__(self, other) -> "Pmf[bool]":
        return self.op(operator.__add__, other)

    def __radd__(self, other) -> "Pmf[bool]":
        return self.op(operator.__add__, other)

    def op(self, op: Callable[[TOutcome, Any], Any], other: Any) -> "Pmf[Any]":
        other = self.coerce(other)
        joint = Joint([self, other])
        return joint.map_outcome(op)

    def bool_op(self, op: Callable[[TOutcome, Any], bool], other: Any) -> "Pmf[bool]":
        bool_pmf = self.op(op, other)
        return bool_pmf.union(TablePmf({True: 0, False: 0}))

    def union(self, other) -> "Pmf[TOutcome]":
        pmf = {}
        for outcome, probability in itertools.chain(self, other):
            pmf[outcome] = pmf.get(outcome, 0) + probability
        return pmf_from_table(pmf)

    @classmethod
    def if_(cls, condition_pmf: "Pmf[bool]", then_pmf: AnyPmf, else_pmf: AnyPmf) -> "Pmf[Any]":
        return cls.coerce(then_pmf).scale_probability(condition_pmf.p(True)) \
            .union(cls.coerce(else_pmf).scale_probability(condition_pmf.p(False)))

    def scale_probability(self, scale: float) -> "Pmf[TOutcome]":
        raise NotImplementedError

    def __getattr__(self, item):
        return self.map_pmf(lambda outcome: getattr(outcome, item))


def pmf_from_table(table: Dict[TOutcome, float]) -> Pmf[TOutcome]:
    outcome = next(iter(table.keys()))
    if isinstance(outcome, int) and not isinstance(outcome, bool):
        return IntegerInterval.from_table(table)
    else:
        return TablePmf(table)


IntDist = Union[int, Pmf[int]]


class PmfStats:
    def __init__(self, mean: float, std: float, percentile80: int, percentile20: int):
        self.mean = mean
        self.std = std
        self.percentile80 = percentile80
        self.percentile20 = percentile20

    def __str__(self):
        return "{:.2f} +/- {:.2f}, 80/20p: {}/{}".format(self.mean, self.std, self.percentile80, self.percentile20)


class IntegerInterval(Pmf[int]):
    def __init__(self, probabilities: List[float], offset: int):
        super().__init__()
        self.probabilities = probabilities
        self.size = len(self.probabilities)
        self.offset = offset
        self.__cdf = None
        self.__mean = None
        self.__std = None

    @classmethod
    def from_table(cls, table: Dict[int, float]) -> "IntegerInterval":
        min_outcome: int = min(table.keys())
        max_outcome = max(table.keys())
        size = max_outcome - min_outcome + 1
        probabilities = [0.0] * size
        for outcome, probability in table.items():
            probabilities[outcome - min_outcome] = probability
        return cls(probabilities, min_outcome)

    def cdf(self):
        if self.__cdf is None:
            self.__cdf = []
            prefix_sum = 0
            for probability in self.probabilities:
                self.__cdf.append(prefix_sum)
                prefix_sum += probability
        return self.__cdf

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        return map(lambda i: (self.offset + i, self.probabilities[i]), range(self.size))

    @staticmethod
    def __add_helper(interval1, interval2):
        probabilities = []
        for s in range(interval1.size):
            probabilities.append(0)
            for i in range(s + 1):
                probabilities[s] += interval1.probabilities[i] * interval2.probabilities[s - i]
        for s in range(interval1.size, interval1.size + interval2.size - 1):
            probabilities.append(0)
            for i in range(s - interval1.size + 1, min(interval2.size, s + 1)):
                probabilities[s] += interval1.probabilities[s - i] * interval2.probabilities[i]
        return IntegerInterval(probabilities, interval1.offset + interval2.offset)

    def __add__(self, other):
        if isinstance(other, int):
            return IntegerInterval(self.probabilities, self.offset + other)
        if self.size <= other.size:
            return self.__add_helper(self, other)
        else:
            return self.__add_helper(other, self)

    def __radd__(self, other):
        return self.__add__(other)

    def adv(self):
        probabilities = []
        cdf = self.cdf()
        for i in range(len(self.probabilities)):
            probabilities.append(cdf[i] * self.probabilities[i] +
                                 self.probabilities[i] * (cdf[i] + self.probabilities[i]))
        return IntegerInterval(probabilities, self.offset)

    def ge(self, threshold: int):
        cdf = self.cdf()
        probabilities = []
        threshold_index = threshold - self.offset
        for i in range(threshold_index):
            probabilities.append(self.probabilities[i] * cdf[threshold_index])
        for i in range(threshold_index, len(self.probabilities)):
            probabilities.append(self.probabilities[i] * (1 + cdf[threshold_index]))
        return IntegerInterval(probabilities, self.offset)

    def times(self, n, op=operator.__add__):
        return functools.reduce(op, itertools.repeat(self, n))

    @staticmethod
    def __union_helper(interval1, interval2):
        probabilities = interval1.probabilities + \
                        [0] * max(0, interval2.offset + interval2.size - interval1.offset - interval1.size)
        for i in range(interval2.size):
            probabilities[interval2.offset - interval1.offset + i] += interval2.probabilities[i]
        return IntegerInterval(probabilities, interval1.offset)

    def union(self, other):
        if not isinstance(other, IntegerInterval):
            return super(self.__class__, self).union(other)
        if self.offset <= other.offset:
            return self.__union_helper(self, other)
        else:
            return self.__union_helper(other, self)

    def scale_probability(self, scale: float):
        return IntegerInterval(list(map(lambda p: p * scale, self.probabilities)), self.offset)

    def p(self, outcome: int) -> float:
        return self.probabilities[outcome - self.offset]

    def outcome_to_str(self, outcome: int) -> str:
        return "{:>3}".format(outcome)

    def stats(self) -> PmfStats:
        return PmfStats(self.mean(), self.std(), self.percentile(0.8), self.percentile(0.2))

    def percentile(self, percentile: float) -> int:
        return self.offset + bisect.bisect(self.cdf(), 1 - percentile)

    def mean(self) -> float:
        if self.__mean is None:
            self.__mean = 0
            for outcome, probability in self:
                self.__mean += outcome * probability
        return self.__mean

    def std(self) -> float:
        if self.__std is None:
            self.__std = 0
            mean = self.mean()
            for outcome, probability in self:
                self.__std += ((outcome - mean) ** 2) * probability
            self.__std = math.sqrt(self.__std)
        return self.__std

    def at_least(self) -> Tuple[Iterable[int], Iterator[float]]:
        cdf = self.cdf()
        return map(lambda i: self.offset + i, range(self.size)), map(lambda i: 1 - cdf[i], range(self.size))


class TablePmf(Pmf[TOutcome]):
    def __init__(self, table: Dict[TOutcome, float]):
        super().__init__()
        self.table = table

    def __iter__(self) -> Iterator[Tuple[TOutcome, float]]:
        return iter(self.table.items())

    def p(self, outcome: TOutcome) -> float:
        return self.table[outcome]

    def scale_probability(self, scale: float):
        return TablePmf({outcome: probability * scale for outcome, probability in self.table.items()})


class Constant(TablePmf[TOutcome]):
    def __init__(self, constant: TOutcome):
        super().__init__({constant: 1})


class Joint(Pmf[Tuple]):
    def __init__(self, pmfs: List[Pmf[Any]]):
        super().__init__()
        self.pmfs = pmfs

    def __iter__(self) -> Iterator[Tuple[Tuple, float]]:
        return map(lambda product: self.aggregate_to_tuple(functools.reduce(
            self.aggregate_outcome_probabilities, product, [[], 1])),
                   itertools.product(*self.pmfs))

    @staticmethod
    def aggregate_outcome_probabilities(aggregated_outcome_probability: List,
                                        outcome_probability: Tuple[Any, float]) -> List:
        aggregated_outcome_probability[0].append(outcome_probability[0])
        aggregated_outcome_probability[1] *= outcome_probability[1]
        return aggregated_outcome_probability

    @staticmethod
    def aggregate_to_tuple(aggregated_outcome_probability: List) -> Tuple[Tuple, float]:
        return tuple(aggregated_outcome_probability[0]), aggregated_outcome_probability[1]

    def map_sub_pmf(self, f: Callable[[Pmf[Any], Any], Any], outcome: Tuple):
        return map(lambda op: f(*op), zip(self.pmfs, outcome))

    def p(self, outcome: Tuple) -> float:
        return functools.reduce(operator.__mul__,
                                self.map_sub_pmf(lambda pmf, pmf_outcome: pmf.p(pmf_outcome), outcome))

    def map_outcome(self, f: Callable) -> Pmf[Any]:
        return self.map_pmf(lambda outcome: f(*outcome))

    def scale_probability(self, scale: float):
        pmfs = self.pmfs[:-1]
        pmfs.append(self.pmfs[-1].scale_probability(scale))
        return Joint(pmfs)


def die(number_of_faces: int):
    return IntegerInterval([1 / number_of_faces] * number_of_faces, 1)


def d(*args):
    if len(args) == 1:
        return die(args[0])
    elif len(args) == 2:
        return functools.reduce(operator.__add__, itertools.repeat(die(args[1]), args[0]))
    else:
        raise ValueError(args)


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


class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class HitOutcome(OrderedEnum):
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


def resolve_hit(attack: Attack, armor_class: Pmf[int]) -> Pmf[HitOutcome]:
    def resolve_hit_helper(roll: int) -> Union[HitOutcome, Pmf[HitOutcome]]:
        if roll >= attack.critical_threshold:
            return HitOutcome.CRITICAL_HIT
        elif roll == 1:
            return HitOutcome.CRITICAL_MISS
        else:
            return Pmf.if_(roll + attack.attack_bonus >= armor_class, HitOutcome.NORMAL_HIT, HitOutcome.NORMAL_MISS)
    hit_outcome = attack.attack_roll.map_nested(resolve_hit_helper)
    return hit_outcome


def resolve_attack(attack: Attack, armor_class: Pmf[int]) -> Pmf[AttackOutcome]:
    def resolve_attack_damage(hit_outcome: HitOutcome) -> Pmf[int]:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return attack.damage_base + attack.damage_base + attack.damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return attack.damage_base + attack.damage_bonus
        else:
            return Pmf.coerce(0)

    def resolve_attack_outcome(hit_outcome: HitOutcome) -> Pmf[AttackOutcome]:
        damage_dist = resolve_attack_damage(hit_outcome)
        return damage_dist.map_pmf(lambda damage: AttackOutcome(hit_outcome, damage))

    hit = resolve_hit(attack, armor_class)
    return hit.map_nested(resolve_attack_outcome)


class AttackBuilder:
    def __init__(self, damage_base: IntDist):
        self.damage_base = damage_base
        self.damage_bonus = 0
        self.is_adv = False
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

    def dmgbon(self, bonus_damage: int):
        self.damage_bonus += bonus_damage
        return self

    def adv(self, is_adv=True):
        self.is_adv = is_adv
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
        self.is_great_weapon_master_or_sharpshooter = is_great_weapon_master_or_sharpshooter
        return self

    def __lucky_roll(self, attack_roll: Pmf[int]):
        return attack_roll.map_nested(lambda roll: attack_roll if roll == 1 else roll)

    def build(self):
        bonus_damage = self.damage_bonus
        if self.add_ability_modifier_to_damage:
            bonus_damage += self.ability_modifier

        attack_roll = d(20)

        if self.is_adv:
            if self.is_lucky:
                attack_roll = attack_roll.map_nested(lambda roll1: attack_roll.adv()
                    if roll1 == 1 else self.__lucky_roll(attack_roll).map_pmf(lambda roll2: max(roll2, roll1)))
            else:
                attack_roll = attack_roll.adv()
        elif self.is_lucky:
            attack_roll = self.__lucky_roll(attack_roll)

        attack_bonus = self.attack_bonus + self.proficiency_bonus + self.ability_modifier

        if self.is_great_weapon_master_or_sharpshooter:
            bonus_damage += 10
            attack_bonus -= 5

        return Attack(Pmf.coerce(self.damage_base),
                      bonus_damage,
                      attack_roll,
                      Pmf.coerce(attack_bonus),
                      self.critical_threshold)

    def times(self, n: int) -> List[Pmf[int]]:
        attack = self.build()
        return [attack for _ in range(n)]

    def resolve(self, armor_class: IntDist) -> Pmf[int]:
        attack = self.build()
        return resolve_attack(attack, Pmf.coerce(armor_class)).map_pmf(lambda outcome: outcome.damage)


def resolve_turn_attacks(*attacks,
                         armor_class: IntDist,
                         extra_damage_roll: IntDist = 0,
                         damage_bonus: int = 0) -> Pmf[int]:
    def resolve_turn_extra_damage(hit_outcome: HitOutcome) -> Pmf[int]:
        if hit_outcome == HitOutcome.CRITICAL_HIT:
            return Pmf.coerce(extra_damage_roll).times(2) + damage_bonus
        elif hit_outcome == HitOutcome.NORMAL_HIT:
            return extra_damage_roll + damage_bonus
        else:
            return Pmf.coerce(0)
    attacks = itertools.chain.from_iterable(map(lambda attack: attack if isinstance(attack, List) else [attack], attacks))
    attacks = map(lambda attack: attack if isinstance(attack, Attack) else attack.build(), attacks)
    attacks = list(map(lambda attack: resolve_attack(attack, armor_class), attacks))
    hardest_hit: Pmf[HitOutcome] = functools.reduce(max, map(lambda attack: attack.hit_outcome, attacks))
    total_damage = sum(map(lambda attack: attack.damage, attacks)) + hardest_hit.map_nested(resolve_turn_extra_damage)
    return total_damage


test_sum = die(1) + die(4)
assert 0.25 == test_sum.p(3)
assert 0.25 == test_sum.p(5)

assert 0.28 == d(5).adv().p(4)
assert 0.08000000000000002 == d(5).ge(3).p(2)
assert 0.27999999999999997 == d(5).ge(3).p(3)

test_joint = Joint([d(2), d(4)])
assert 0.125 == test_joint.p((1, 2))

assert 0.25 == (d(4) == d(4)).p(True)

test_union = TablePmf({"a": 0.1, "b": 0.2}).union(TablePmf({"b": 0.4, "c": 0.3}))
assert 0.6000000000000001 == test_union.p("b")
assert 0.1 == test_union.p("a")

test_union = d(2).union(d(2) + d(1))
assert 1.0 == test_union.p(2)

test_scale = d(5).scale_probability(0.5)
assert 0.1 == test_scale.p(2)

test_if = Pmf.if_(d(5) == 1, d(1), d(1) + d(1))
assert 0.2 == test_if.p(1)
assert 0.8 == test_if.p(2)

assert (1 + d(3)).p(3) == (d(3) + 1).p(3)


test_attack0 = AttackBuilder(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2).build()
assert 0.6525 == resolve_hit(test_attack0, Pmf.coerce(15))

test_attack1 = AttackBuilder(d(1)).adv().lucky().build()
hit_outcome1 = resolve_hit(test_attack1, 15)
assert 0.10212500000000002 == hit_outcome1.p(HitOutcome.CRITICAL_HIT)

test_damage0 = resolve_turn_attacks(AttackBuilder(d(10)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2).times(2),
                                    AttackBuilder(d(4)).prof(3).amod(3).adv().gwm().attbon(3).dmgbon(2),
                                    extra_damage_roll=d(2, 8),
                                    armor_class=15)
test_damage0_stats = test_damage0.stats()
assert 52.81875000000001 == test_damage0_stats.mean
assert 0.04861111111111111 == AttackBuilder(d(6)).resolve(15).p(6)
assert 1.0000000000000002 == AttackBuilder(d(1)).attbon(-20).crit(0).resolve(0).p(2)

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
