from functools import *
from itertools import *
from math import sqrt
# from pprint import pprint
import operator


# from multipledispatch import dispatch

class Pmf(object):
    """Probability mass function"""

    def __init__(self):
        super(Pmf, self).__init__()

    def prob(self, outcome):
        raise NotImplementedError

    def num_outcomes(self):
        raise NotImplementedError

    def entries(self):
        raise NotImplementedError

    def map(self, f):
        table = {}
        for outcome, prob in self:
            mapped_outcome = f(outcome)
            # print(outcome, prob, mapped_outcome)
            table[mapped_outcome] = table.get(mapped_outcome, 0) + prob
        return TablePmf(table)

    """
    @classmethod
    def reduce(cls, op, pmfs, unit = None):
        def pmf_op(pmf1, pmf2):
            return cls.multimap(op, (pmf1, pmf2))
        if unit == None:
            return reduce(pmf_op, pmfs)
        else:
            return reduce(pmf_op, pmfs, unit)
    """

    @classmethod
    def multimap(cls, f, pmfs):
        return JointPmf(pmfs).map(lambda p: f(*p))

    @classmethod
    def multimap_any(cls, f, objs):
        return cls.multimap(f, tuple(map(lambda obj: cls.any_to_value_pmf(obj), objs)))

    @classmethod
    def ifthenelse(cls, condition_pmf, then_pmf, else_pmf):
        return cls.multimap_any(lambda oc, ot, oe: ot if oc else oe, (condition_pmf, then_pmf, else_pmf))

    @classmethod
    def maketuple(cls, pmfs):
        return cls.multimap_any(lambda *os: os, pmfs)

    def to_dict(self):
        return {outcome: prob for outcome, prob in iter(self)}

    def at_least(self):
        at_least_probs = {}
        probs = self.to_dict()
        accum = 0
        for outcome in reversed(sorted(probs)):
            at_least_probs[outcome] = probs[outcome] + accum
            accum += probs[outcome]
        return at_least_probs

    @staticmethod
    def almost_equal(n1, n2, tolerance=0.0001):
        return abs((n1 - n2) / n2) < tolerance

    @staticmethod
    def dict_almost_equal(dict1, dict2, tolerance=0.0001):
        if len(dict1) != len(dict2):
            return False
        for key, value in dict1.items():
            if not Pmf.almost_equal(value, dict2[key], tolerance):
                print(key, value, dict2[key])
                return False
        return True

    def is_equiv(self, other, tolerance=0.0001):
        # print(self, other)
        if len(self) != len(other):
            return False
        for outcome, prob in iter(self):
            if not Pmf.almost_equal(prob, other[outcome], tolerance):
                print(outcome, prob, other[outcome])
                return False
        return True

    def to_value_pmf(self):
        return self

    @classmethod
    def any_to_value_pmf(cls, obj):
        if isinstance(obj, Pmf):
            return obj.to_value_pmf()
        else:
            return ConstantPmf(obj)

    def pmf_op(self, other, op):
        # return self.multimap(op, (self.to_value_pmf(), self.any_to_value_pmf(other)))
        return self.multimap_any(op, (self, other))

    def avg(self):
        return reduce(operator.add, map(lambda i: i[0] * i[1], iter(self.to_value_pmf())))

    def sd(self):
        return sqrt(self.to_value_pmf().map(lambda n: n ** 2).avg() - (self.avg()) ** 2)

    def print_stats(self):
        print(self.avg(), self.sd(), self.at_least())

    def __getitem__(self, key):
        return self.prob(key)

    def __len__(self):
        return self.num_outcomes()

    def __iter__(self):
        return self.entries()

    def __radd__(self, other):
        return self.pmf_op(other, operator.add)

    def __add__(self, other):
        return self.pmf_op(other, operator.add)

    def __rsub__(self, other):
        return self.pmf_op(other, operator.sub)

    def __sub__(self, other):
        return self.pmf_op(other, operator.sub)

    def __neg__(self):
        return self.map(operator.neg)

    def __rmul__(self, other):
        return self.pmf_op(other, operator.mul)

    def __mul__(self, other):
        return self.pmf_op(other, operator.mul)

    def __rtruediv__(self, other):
        return self.pmf_op(other, operator.truediv)

    def __truediv__(self, other):
        return self.pmf_op(other, operator.truediv)

    def __rfloordiv__(self, other):
        return self.pmf_op(other, operator.floordiv)

    def __floordiv__(self, other):
        return self.pmf_op(other, operator.floordiv)

    def __lt__(self, other):
        return self.pmf_op(other, operator.lt)

    def __le__(self, other):
        return self.pmf_op(other, operator.le)

    def __eq__(self, other):
        return self.is_equiv(other)

    def __ge__(self, other):
        return self.pmf_op(other, operator.ge)

    def __gt__(self, other):
        return self.pmf_op(other, operator.gt)

    def __not__(self):
        return self.map(operator.not_)

    def __and__(self, other):
        return self.pmf_op(other, operator.and_)

    def __or__(self, other):
        return self.pmf_op(other, operator.or_)

    def __str__(self):
        return str(self.to_dict())


# @dispatch(Pmf, Pmf, object)
# def pmf_op(pmf1, pmf2, op):
# return Pmf.multimap(op, (pmf1, pmf2))#JointPmf((pmf1, pmf2)).map(lambda p: op(*p))

"""
@dispatch(Pmf, int, object)
def pmf_op(pmf, c, op):
	return self.pmf_op(ConstantPmf(c), op)
"""


class TablePmf(Pmf):
    """Table based Pmf"""

    def __init__(self, table):
        super(TablePmf, self).__init__()
        self.table = table

    def prob(self, outcome):
        return self.table.get(outcome, 0)

    def num_outcomes(self):
        return len(self.table)

    def entries(self):
        return iter(self.table.items())


class ConstantPmf(TablePmf):
    def __init__(self, n):
        super(ConstantPmf, self).__init__({n: 1})


class D(Pmf):
    def __init__(self, faces):
        super(D, self).__init__()
        self.outcomes = range(1, faces + 1)
        self.p = 1 / faces

    def prob(self, outcome):
        if outcome in self.outcomes:
            return self.p
        else:
            return 0

    def num_outcomes(self):
        return len(self.outcomes)

    def entries(self):
        for outcome in self.outcomes:
            yield (outcome, self.p)


class JointPmf(Pmf):
    """Joint Pmf"""

    def __init__(self, pmfs):
        super(JointPmf, self).__init__()

        # print(1)
        # print(pmfs)
        # for p in pmfs: print(p)
        # print(2)

        self.pmfs = pmfs  # ((pmf if isinstance(pmf, Pmf) else ConstantPmf(pmf)) for pmf in pmfs)

    def prob(self, outcome):
        return reduce(operator.mul, imap(lambda pmf, o: pmf[o], self.pmfs, outcome))

    def num_outcomes(self):
        return reduce(operator.mul, map(lambda pmf: pmf.num_outcomes(), self.pmfs))

    def entries(self):
        return self._entries_helper(iter(self.pmfs))

    def _entries_helper(self, pmfs_iter):
        try:
            pmf = next(pmfs_iter)
            for suffix_outcome, suffix_prob in self._entries_helper(pmfs_iter):
                for outcome, prob in pmf:
                    yield ((outcome,) + suffix_outcome, prob * suffix_prob)
        except StopIteration:
            yield ((), 1)

    def sum(self):
        return self.map(sum)

    def reduce(self, op, unit=None):
        def pmf_op(pmf1, pmf2):
            return Pmf.multimap(op, (pmf1, pmf2))

        # print("r", self.pmfs)
        if unit == None:
            return reduce(pmf_op, self.pmfs)
        else:
            return reduce(pmf_op, self.pmfs, unit)

    def to_value_pmf(self):
        return self.sum()

    def largest_n(self, n):
        # print("l", self.pmfs)
        return JointPmf(
            tuple(map(lambda pmf: pmf.map(lambda n: (n,) + tuple((float('-inf') for _ in range(n - 1)))),
                      self.pmfs))).reduce(lambda t1, t2: tuple(sorted(t1 + t2)[-n:]))

    def nth_largest(self, n):
        return self.largest_n(n).map(lambda t: t[-n])

    def max(self):
        return self.reduce(max)

    """def __radd__(self, other):
        return self.sum().__radd__(other)

    def __add__(self, other):
        return self.sum().__add__(other)

    def __rsub__(self, other):
        return self.sum().__rsub__(other)

    def __sub__(self, other):
        return self.sum().__sub__(other)

    def __neg__(self):
        return self.sum().__neg__()

    def __rmul__(self, other):
        return self.sum().__rmul__(other)

    def __mul__(self, other):
        return self.sum().__mul__(other)

    def __rtruediv__(self, other):
        return self.sum().__rtruediv__(other)

    def __truediv__(self, other):
        return self.sum().__truediv__(other)

    def __rfloordiv__(self, other):
        return self.sum().__rfloordiv__(other)

    def __floordiv__(self, other):
        return self.sum().__floordiv__(other)

    def __lt__(self, other):
        return other > self.sum()

    def __le__(self, other):
        return other >= self.sum()

    def __ge__(self, other):
        return other <= self.sum()

    def __gt__(self, other):
        return other < self.sum()
        #return self.sum().__gt__(other)
"""

    def __not__(self):
        raise TypeError

    def __and__(self, other):
        raise TypeError

    def __or__(self, other):
        raise TypeError


class NPmf(JointPmf):
    """N many Pmfs"""

    def __init__(self, n, pmf):
        super(NPmf, self).__init__(tuple(pmf for _ in range(n)))


class ND(NPmf):
    """N dice"""

    def __init__(self, n, faces):
        super(ND, self).__init__(n, D(faces))


# print(ND(3, 4) > D(15))
# print(ND(3, 4) > 8)
# print(D(4) + ND(2, 4))
assert D(4) + ND(2, 4) == ND(3, 4).sum()

# exit()

assert D(2) + D(2) == TablePmf({2: 0.25, 3: 0.5, 4: 0.25})

assert D(2) + 1 == TablePmf({2: 0.5, 3: 0.5})

assert 1 + D(2) == TablePmf({2: 0.5, 3: 0.5})

assert ND(2, 2) + 1 == TablePmf({3: 0.25, 4: 0.5, 5: 0.25})

"""
roll = D(20)


if roll > 15:
	x = roll
	condition roll > 15
	hit = true
else:
	condition roll <= 15
	hit = false
merge branch
hit


"""

stat_roll = ND(4, 6).largest_n(3).map(sum)

assert stat_roll == TablePmf(
    {3: 0.0007716049382716049, 4: 0.0030864197530864196, 5: 0.007716049382716049, 6: 0.016203703703703703,
     7: 0.029320987654320986, 8: 0.047839506172839504, 9: 0.07021604938271606, 10: 0.09413580246913579,
     11: 0.11419753086419754, 12: 0.12885802469135801, 13: 0.13271604938271603, 14: 0.12345679012345678,
     15: 0.10108024691358025, 16: 0.07253086419753085, 17: 0.04166666666666666, 18: 0.016203703703703703})
assert Pmf.dict_almost_equal(NPmf(6, stat_roll).nth_largest(2).at_least(),
                             {18: 0.0037712978997010613, 17: 0.04297174396189502, 16: 0.1785025147554727,
                              15: 0.42163254105952985, 14: 0.6901003676248133, 13: 0.8786176420382443,
                              12: 0.9661397862008139, 11: 0.9934099842270956, 10: 0.9991552411369676,
                              9: 0.9999303248135635, 8: 0.9999965317348086, 7: 0.9999999030679377, 6: 0.999999998765834,
                              5: 0.9999999999948876, 4: 0.9999999999999979, 3: 0.9999999999999996})
assert Pmf.dict_almost_equal(NPmf(6, stat_roll).nth_largest(1).at_least(),
                             {18: 0.09336788352756517, 17: 0.30069928149459824, 16: 0.5675723186031841,
                              15: 0.7939721069010339, 14: 0.9279543680017638, 13: 0.9819125032121617,
                              12: 0.9968194053566061, 11: 0.9996186510521065, 10: 0.9999711247911593,
                              9: 0.9999986646243858, 8: 0.999999965345446, 7: 0.9999999995406059, 6: 0.9999999999975957,
                              5: 0.9999999999999963, 4: 0.9999999999999997, 3: 0.9999999999999997})
assert Pmf.dict_almost_equal(NPmf(6, stat_roll).largest_n(2).map(sum).at_least(),
                             {36: 0.0037712978997010613, 35: 0.021204173018408268, 34: 0.06678950251364398,
                              33: 0.1503173497095011, 32: 0.27328695484648013, 31: 0.4195516640008528,
                              30: 0.5731467072301601, 29: 0.7086608028822727, 28: 0.819832313566671,
                              27: 0.8961149664818756, 26: 0.9467231694762607, 25: 0.9741534736268778,
                              24: 0.9891492615027956, 23: 0.9955834539859781, 22: 0.9985102911733038,
                              21: 0.9994953600444297, 20: 0.9998688100006322, 19: 0.9999635085211747,
                              18: 0.9999929491919841, 17: 0.999998399446, 16: 0.9999997840507013,
                              15: 0.9999999605103931, 14: 0.9999999965711357, 13: 0.9999999995038211,
                              12: 0.9999999999772524, 11: 0.9999999999975125, 10: 0.9999999999999567,
                              9: 0.9999999999999963, 8: 0.9999999999999997, 7: 0.9999999999999997,
                              6: 0.9999999999999997})
assert ND(2, 10).max() == TablePmf(
    {1: 0.010000000000000002, 2: 0.030000000000000006, 3: 0.05000000000000001, 4: 0.07, 5: 0.09000000000000002,
     6: 0.11000000000000004, 7: 0.13000000000000006, 8: 0.15000000000000008, 9: 0.1700000000000001,
     10: 0.1900000000000001})

assert D(6).avg() == 3.5
assert ND(2, 6).avg() == 7
assert Pmf.almost_equal(D(6).sd(), 1.707825127659933)
assert Pmf.almost_equal(ND(2, 6).sd(), 2.4152294576982403)
normal = ND(2, 6).sum()
savage = NPmf(2, normal).max()
normal.print_stats()
savage.print_stats()

ATTACK_HIT_CRIT = 2
ATTACK_HIT_NON_CRIT = 1
ATTACK_MISS = 0

"""def attack_basic(attack_base, attack_bonus, ac, damage_roll, damage_bonus, critical_threshold = 20):
	#print("a1", damage_roll * 2 + damage_bonus)
	return Pmf.ifthenelse(attack_base >= critical_threshold,
		Pmf.maketuple((damage_roll * 2 + damage_bonus, ATTACK_HIT_CRIT)),
		Pmf.ifthenelse(attack_base > 1 and attack_base + attack_bonus >= ac,
			Pmf.maketuple((damage_roll + damage_bonus, ATTACK_HIT_NON_CRIT)),
			Pmf.maketuple((0, ATTACK_MISS))
		)
	)"""
# attack_base > 1 and
"""def attack_basic(attack_base, attack_bonus, ac, damage_roll, damage_bonus, critical_threshold = 20):
	#print("a1", damage_roll * 2 + damage_bonus)
	return Pmf.ifthenelse(attack_base >= critical_threshold,
		ATTACK_HIT_CRIT,
		Pmf.ifthenelse(attack_base < critical_threshold and attack_base + attack_bonus >= ac,
			ATTACK_HIT_NON_CRIT,
			ATTACK_MISS
		)
	)
"""


# print(D(20) <= 8)
# print(attack_basic(D(20), 9, 18, ND(2, 6), 5))

def d(t):
    return {o: 1 / t for o in range(1, t + 1)}


def npmf(n, pmf):
    return tuple((pmf for _ in range(n)))


# @infix
def nd(n, t):
    return npmf(n, d(t))


pmf_empty = {(): 1}


def n(x):
    return {x: 1}


def pmf_map(f, pmf):
    npmf = {}
    for o, p in pmf.items():
        no = f(o)
        # print(no)
        npmf[no] = npmf.get(no, 0) + p
    # print(npmf)
    return npmf


def join(*pmfs):
    jpmf = pmf_empty
    for pmf in pmfs:
        jpmf = {jo + (o,): jp * p for jo, jp in jpmf.items() for o, p in pmf.items()}
    return jpmf


def pmf_reduce(op, pmfs, u=None):
    def pmf_op(pmf1, pmf2):
        # print(pmf1)
        # print(pmf2)
        pmf = join(pmf1, pmf2)
        # print(pmf)
        # return pmf_map(lambda o: op(*o), pmf)
        return pmf_map(lambda o: op(*o), pmf)

    if u == None:
        return reduce(pmf_op, pmfs)
    else:
        return reduce(pmf_op, pmfs, u)


def pmf_func(f, *pmfs):
    return pmf_map(lambda o: f(*o), join(*pmfs))


def pmf_if(c, pmf_c, pmf_t, pmf_f):
    return pmf_func(lambda oc, ot, of: ot if c(oc) else of, pmf_c, pmf_t, pmf_f)


def pmf_sum(*pmfs):
    return pmf_reduce(lambda o1, o2: o1 + o2, pmfs)


def pmf_subtract(pmf1, pmf2):
    return pmf_func(lambda o1, o2: o1 - o2, pmf1, pmf2)


def pmf_max(*pmfs):
    return pmf_reduce(max, pmfs)


def pmf_min(*pmfs):
    return pmf_reduce(min, pmfs)


def npmf_sum(n, pmf):
    return pmf_sum(*npmf(n, pmf))


def at_least(pmf):
    d = {}
    a = 0
    for o in reversed(sorted(pmf)):
        d[o] = pmf[o] + a
        a += pmf[o]
    return d


def top_nth(n, *pmfs):
    return pmf_reduce(
        lambda o1, o2: tuple(sorted(o1 + o2)[-n:]),
        (pmf_map(lambda o: (o,) + tuple((float('-inf') for _ in range(n - 1))), pmf) for pmf in pmfs)
    )


def nth_largest(n, *pmfs):
    return pmf_map(
        lambda o: o[-n],
        top_nth(n, *pmfs),
    )


def attack_basic(attack_base, attack_bonus, ac, damage_roll, damage_bonus, critical_threshold=20):
    # print(attack_base, attack_bonus)
    if attack_base >= critical_threshold:
        return (damage_roll * 2 + damage_bonus, ATTACK_HIT_CRIT)
    elif attack_base > 1 and attack_base + attack_bonus >= ac:
        return (damage_roll + damage_bonus, ATTACK_HIT_NON_CRIT)
    else:
        return (0, ATTACK_MISS)


PROFICIENCY_BONUS = 4
ABILITY_MODIFIER = 5
CHARISMA_MODIFIER = 3
AC = n(18)


def attack(
        damage_base,
        proficiency_bonus=PROFICIENCY_BONUS,
        ability_modifier=ABILITY_MODIFIER,
        add_ability_modifier_to_damange=True,
        is_attack_adv=False,
        is_archery=False,
        is_great_weapon_fighting=False,
        is_blessed=False,
        is_sacred_weapon=False,
        critical_threshold=20):
    attack_bonus = n(proficiency_bonus + ability_modifier)
    damage_bonus = 0
    if is_attack_adv:
        attack_base = pmf_max(*nd(2, 20))
    else:
        attack_base = d(20)
    if add_ability_modifier_to_damange:
        damage_bonus += ability_modifier
    if is_great_weapon_fighting:
        attack_bonus = pmf_subtract(attack_bonus, n(5))
        damage_bonus += 10
    if is_blessed:
        attack_bonus = pmf_sum(attack_bonus, d(4))
    if is_sacred_weapon:
        attack_bonus += CHARISMA_MODIFIER
    return pmf_func(lambda aba, abo, ac, db: attack_basic(
        attack_base=aba,
        attack_bonus=abo,
        ac=ac,
        damage_roll=db,
        damage_bonus=damage_bonus,
        critical_threshold=critical_threshold),
                    attack_base, attack_bonus, AC, damage_base)


"""print(pmf_func(lambda aba, abo, ac, db: attack_basic(
			attack_base=aba,
			attack_bonus=abo,
			ac=ac,
			damage_roll=db,
			damage_bonus=5),
		d(20), n(9), n(18), pmf_sum(*nd(2, 6))))
"""


def multi_attack_damage(*attack_pmfs, extra_damage_roll=n(0), extra_damange_bonus=0):
    # print(attack_pmfs[0])
    all_attacks_pmf = pmf_reduce(lambda a1, a2: (a1[0] + a2[0], max(a1[1], a2[1])), attack_pmfs)
    # print(all_attacks_pmf)
    return pmf_func(lambda aa, ed: aa[0] + aa[1] * (ed + extra_damange_bonus), all_attacks_pmf, extra_damage_roll)


def damage_against_save(save_dc, save_bonus, damage, is_save_disadvantage=False, is_save_half=False):
    if is_save_disadvantage:
        save_roll = pmf_min(*nd(2, 20))
    else:
        save_roll = d(20)
    if is_save_half:
        save_success_damage = pmf_map(lambda o: o // 2, damage)
    else:
        save_success_damage = n(0)
    return pmf_if(lambda s: s + save_bonus < save_dc, save_roll, damage, save_success_damage)


def pmf_equal(pmf1, pmf2, tolerance=0.0001):
    if len(pmf1) != len(pmf2):
        return False
    for o, p in pmf1.items():
        if o not in pmf2 or abs(pmf2[o] - p) / pmf2[o] > tolerance:
            print(o, pmf2[o], p)
            return False
    return True


# print(at_least(damage_against_save(save_dc=17, save_bonus=7, damage=pmf_sum(n(0), *nd(3, 12)), is_save_half=True)))


assert at_least(multi_attack_damage(*npmf(6, attack(pmf_sum(d(10), d(6)))))) == {222: 3.3489797668038423e-19,
                                                                                 220: 4.3536736968449956e-18,
                                                                                 218: 3.0475715877914974e-17,
                                                                                 216: 1.5237857938957488e-16,
                                                                                 214: 6.095143175582995e-16,
                                                                                 212: 2.0723486796982186e-15,
                                                                                 210: 6.215036651234575e-15,
                                                                                 208: 1.684871720679014e-14,
                                                                                 206: 4.202634709362144e-14,
                                                                                 205: 4.207055362654325e-14,
                                                                                 204: 9.787594307270242e-14,
                                                                                 203: 9.840642146776415e-14,
                                                                                 202: 2.1577476637517166e-13,
                                                                                 201: 2.1922287594307288e-13,
                                                                                 200: 4.564170471107685e-13,
                                                                                 199: 4.724640185613857e-13,
                                                                                 198: 9.381815173825455e-13,
                                                                                 197: 9.979929564257554e-13,
                                                                                 196: 1.894035266096538e-12,
                                                                                 195: 2.0836370857981843e-12,
                                                                                 194: 3.782116381065675e-12,
                                                                                 193: 4.312373743462794e-12,
                                                                                 192: 7.490725670331794e-12,
                                                                                 191: 8.831111955054018e-12,
                                                                                 190: 1.469895351080248e-11,
                                                                                 189: 1.7816221386316884e-11,
                                                                                 188: 2.8485785590277794e-11,
                                                                                 187: 3.525517618312759e-11,
                                                                                 186: 5.43450946153764e-11,
                                                                                 185: 6.826147996774266e-11,
                                                                                 184: 1.0190801658682703e-10,
                                                                                 183: 1.293492741421254e-10,
                                                                                 182: 1.8796479284550764e-10,
                                                                                 181: 2.404885196973595e-10,
                                                                                 180: 3.4189264108581974e-10,
                                                                                 179: 4.4041658093278483e-10,
                                                                                 178: 6.153987295310359e-10,
                                                                                 177: 7.976379069680216e-10,
                                                                                 176: 1.0995802395458254e-09,
                                                                                 175: 1.4327500689889838e-09,
                                                                                 174: 1.953581400637111e-09,
                                                                                 173: 2.5550641922394777e-09,
                                                                                 172: 3.4505440521449563e-09,
                                                                                 171: 4.520119764539933e-09,
                                                                                 170: 6.048663293909147e-09,
                                                                                 169: 7.917773062749169e-09,
                                                                                 168: 1.0499883226099006e-08,
                                                                                 167: 1.370771379853342e-08,
                                                                                 166: 1.801930551905168e-08,
                                                                                 165: 2.3434061276617696e-08,
                                                                                 164: 3.0558302945695635e-08,
                                                                                 163: 3.957628425035099e-08,
                                                                                 162: 5.125581097621423e-08,
                                                                                 161: 6.613064980321398e-08,
                                                                                 160: 8.518360590009864e-08,
                                                                                 159: 1.0955997715326007e-07,
                                                                                 158: 1.405441421049142e-07,
                                                                                 157: 1.8028761197615264e-07,
                                                                                 156: 2.30509771015183e-07,
                                                                                 155: 2.94932555347584e-07,
                                                                                 154: 3.759062169325088e-07,
                                                                                 153: 4.794983995644319e-07,
                                                                                 152: 6.089779759292081e-07,
                                                                                 151: 7.738079701341335e-07,
                                                                                 150: 9.78627203739538e-07,
                                                                                 149: 1.2377360821514788e-06,
                                                                                 148: 1.5579160736276556e-06,
                                                                                 147: 1.960349573899593e-06,
                                                                                 146: 2.4553414604947656e-06,
                                                                                 145: 3.0739267436898533e-06,
                                                                                 144: 3.832354066045901e-06,
                                                                                 143: 4.775816119953758e-06,
                                                                                 142: 5.930493200215743e-06,
                                                                                 141: 7.361758400603958e-06,
                                                                                 140: 9.111773711737575e-06,
                                                                                 139: 1.1274013331472979e-05,
                                                                                 138: 1.3915037547681439e-05,
                                                                                 137: 1.7166086645201052e-05,
                                                                                 136: 2.1129024563820847e-05,
                                                                                 135: 2.598395582261527e-05,
                                                                                 134: 3.1881775105847536e-05,
                                                                                 133: 3.906413077198373e-05,
                                                                                 132: 4.7748903990092734e-05,
                                                                                 131: 5.825511076692209e-05,
                                                                                 130: 7.089317049781984e-05,
                                                                                 129: 8.608211897103202e-05,
                                                                                 128: 0.00010426368322232004,
                                                                                 127: 0.00012599198892655422,
                                                                                 126: 0.00015189877711226186,
                                                                                 125: 0.00018272625240665325,
                                                                                 124: 0.0002193795566208962,
                                                                                 123: 0.00026286147044929717,
                                                                                 122: 0.00031445682042577225,
                                                                                 121: 0.0003755142978165058,
                                                                                 120: 0.0004478158434089597,
                                                                                 119: 0.0005331385730093929,
                                                                                 118: 0.0006338710810288649,
                                                                                 117: 0.0007522808602403247,
                                                                                 116: 0.0008914505200766179,
                                                                                 115: 0.0010541761653392891,
                                                                                 114: 0.0012443078959162314,
                                                                                 113: 0.0014652119846524472,
                                                                                 112: 0.0017216077772940568,
                                                                                 111: 0.002017539041216797,
                                                                                 110: 0.002358791848553119,
                                                                                 109: 0.002750336522782816,
                                                                                 108: 0.0031993763767807397,
                                                                                 107: 0.0037122415110301286,
                                                                                 106: 0.004298044960182538,
                                                                                 105: 0.004965037380168065,
                                                                                 104: 0.005724736764713471,
                                                                                 103: 0.006587807620014312,
                                                                                 102: 0.007568398785385817,
                                                                                 101: 0.00867971945387806,
                                                                                 100: 0.009938197909649778,
                                                                                 99: 0.01135912744384013,
                                                                                 98: 0.012960168069877435,
                                                                                 97: 0.014757702669013209,
                                                                                 96: 0.01676913244746984,
                                                                                 95: 0.019010898812807514,
                                                                                 94: 0.021498876381575473,
                                                                                 93: 0.024249267701011077,
                                                                                 92: 0.027276167706109473,
                                                                                 91: 0.030596665421207488,
                                                                                 90: 0.03422432411331296,
                                                                                 89: 0.038179713540655295,
                                                                                 88: 0.0424782126956378,
                                                                                 87: 0.047146945699880594,
                                                                                 86: 0.05220507655449236,
                                                                                 85: 0.057687824241469156,
                                                                                 84: 0.06361753574352523,
                                                                                 83: 0.07003537356099147,
                                                                                 82: 0.07696220277604915,
                                                                                 81: 0.08443865488460411,
                                                                                 80: 0.09247599181630142,
                                                                                 79: 0.10110574281767015,
                                                                                 78: 0.11032188071791486,
                                                                                 77: 0.12014079812267253,
                                                                                 76: 0.13053662146814138,
                                                                                 75: 0.14151150383807942,
                                                                                 74: 0.15302620743663828,
                                                                                 73: 0.16507751746572974,
                                                                                 72: 0.17762717928100047,
                                                                                 71: 0.19067988082293333,
                                                                                 70: 0.20421502361061133,
                                                                                 69: 0.21825480312528056,
                                                                                 68: 0.23280495624969533,
                                                                                 67: 0.24790114745639263,
                                                                                 66: 0.2635684968163977,
                                                                                 65: 0.2798341501304599,
                                                                                 64: 0.29671934822880924,
                                                                                 63: 0.31420861788025733,
                                                                                 62: 0.3322937546566702,
                                                                                 61: 0.3508914971250305,
                                                                                 60: 0.3699549402124897,
                                                                                 59: 0.3893325016432111,
                                                                                 58: 0.4089647883711039,
                                                                                 57: 0.42866353434923893,
                                                                                 56: 0.4484001950710131,
                                                                                 55: 0.4679944953379022,
                                                                                 54: 0.48749538158475125,
                                                                                 53: 0.5067712708426519,
                                                                                 52: 0.5259638545070847,
                                                                                 51: 0.5449920070408331,
                                                                                 50: 0.5640658966755194,
                                                                                 49: 0.5831167055354866,
                                                                                 48: 0.6023500899743012,
                                                                                 47: 0.6216381879941681,
                                                                                 46: 0.6411023599755987,
                                                                                 45: 0.6605103369385857,
                                                                                 44: 0.6798613333655006,
                                                                                 43: 0.6988171472939357,
                                                                                 42: 0.7173121816674168,
                                                                                 41: 0.7350280002123484,
                                                                                 40: 0.7519377448973812,
                                                                                 39: 0.7678241411760898,
                                                                                 38: 0.7827733266043884,
                                                                                 37: 0.7968050725795138,
                                                                                 36: 0.8100143093064718,
                                                                                 35: 0.8226125220914361,
                                                                                 34: 0.8346429584849546,
                                                                                 33: 0.8464910173368065,
                                                                                 32: 0.8580443519525474,
                                                                                 31: 0.8696677468414362,
                                                                                 30: 0.8809902445266214,
                                                                                 29: 0.8923598312025474,
                                                                                 28: 0.9031346881481492,
                                                                                 27: 0.91360192888889,
                                                                                 26: 0.9229606266666678,
                                                                                 25: 0.9316321044444456,
                                                                                 24: 0.9388921251851863,
                                                                                 23: 0.945251231851853,
                                                                                 22: 0.9501086140740752,
                                                                                 21: 0.9547112000000011,
                                                                                 20: 0.9585666666666678,
                                                                                 19: 0.9624173333333345,
                                                                                 18: 0.9658290666666678,
                                                                                 17: 0.9695581333333345,
                                                                                 16: 0.9732658666666678,
                                                                                 15: 0.9769789333333345,
                                                                                 14: 0.9803904000000012,
                                                                                 13: 0.9839232000000012,
                                                                                 12: 0.9873024000000012,
                                                                                 11: 0.9902208000000012,
                                                                                 10: 0.9924736000000013,
                                                                                 9: 0.9942144000000013,
                                                                                 8: 0.9953408000000012,
                                                                                 7: 0.9959040000000012,
                                                                                 0: 1.0000000000000013}

ss = pmf_map(sum, top_nth(3, *nd(4, 6)))

assert pmf_if(lambda c: c > 15, d(20), n(1), n(0)) == {0: 0.7500000000000001, 1: 0.25}
