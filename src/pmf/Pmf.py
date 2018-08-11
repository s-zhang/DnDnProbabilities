from typing import *
import functools
import itertools
import operator

TOutcome = TypeVar("TOutcome")


class Pmf(Generic[TOutcome]):
    def __init__(self, pmf_factory: "IPmfFactory"):
        self._pmf_factory = pmf_factory

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

    def coerce(self, obj: Any) -> "Pmf[Any]":
        return self._pmf_factory.from_object(obj)

    def map(self, f: Callable[[TOutcome, float], Any]):
        return map(lambda outcome_probability: f(*outcome_probability), self)

    def map_pmf(self, f: Callable[[TOutcome], Any]) -> "Pmf[Any]":
        pmf = {}
        for outcome, probability in self.map(lambda o, p: (f(o), p)):
            pmf[outcome] = pmf.get(outcome, 0) + probability
        return self._pmf_factory.from_table(pmf)

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
        joint = self._pmf_factory.joint([self, other])
        return joint.map_pmf(lambda outcome: op(*outcome))

    def bool_op(self, op: Callable[[TOutcome, Any], bool], other: Any) -> "Pmf[bool]":
        bool_pmf = self.op(op, other)
        return bool_pmf.union(self._pmf_factory.from_table({True: 0, False: 0}))

    def union(self, other) -> "Pmf[TOutcome]":
        pmf = {}
        for outcome, probability in itertools.chain(self, other):
            pmf[outcome] = pmf.get(outcome, 0) + probability
        return self._pmf_factory.from_table(pmf)

    def scale_probability(self, scale: float) -> "Pmf[TOutcome]":
        raise NotImplementedError

    def __getattr__(self, item):
        return self.map_pmf(lambda outcome: getattr(outcome, item))


class IPmfFactory:
    def from_object(self, obj: Any) -> Pmf[Any]:
        raise NotImplementedError

    def from_table(self, table: Dict[TOutcome, float]) -> Pmf[TOutcome]:
        raise NotImplementedError

    def joint(self, pmfs: List[Pmf[TOutcome]]) -> Pmf[Tuple]:
        raise NotImplementedError

    def ints(self, probabilities: List[float], offset: int) -> Pmf[int]:
        raise NotImplementedError

    def table(self, table: Dict[TOutcome, float]) -> Pmf[TOutcome]:
        raise NotImplementedError

    def const(self, constant: TOutcome) -> Pmf[TOutcome]:
        raise NotImplementedError
