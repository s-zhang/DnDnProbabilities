from typing import *
import functools
import operator
import itertools
from .Pmf import Pmf, IPmfFactory

TOutcome = TypeVar("TOutcome")


class JointPmf(Pmf[Tuple]):
    def __init__(self, pmfs: List[Pmf[Any]], pmf_factory: IPmfFactory):
        super().__init__(pmf_factory)
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

    def scale_probability(self, scale: float):
        pmfs = self.pmfs[:-1]
        pmfs.append(self.pmfs[-1].scale_probability(scale))
        return self._pmf_factory.joint(pmfs)
