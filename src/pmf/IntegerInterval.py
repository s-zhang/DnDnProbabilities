from typing import TypeVar, List, Dict, Iterator, Tuple, Iterable
import functools
import itertools
import operator
import math
import bisect
from .Pmf import Pmf, IPmfFactory

TOutcome = TypeVar("TOutcome")


class PmfStats:
    def __init__(self,
                 mean: float,
                 std: float,
                 percentile80: int,
                 percentile20: int):
        self.mean = mean
        self.std = std
        self.percentile80 = percentile80
        self.percentile20 = percentile20

    def __str__(self):
        return "{:.2f} +/- {:.2f}, 80/20p: {}/{}".format(self.mean,
                                                         self.std,
                                                         self.percentile80,
                                                         self.percentile20)


class IntegerInterval(Pmf[int]):
    def __init__(self,
                 probabilities: List[float],
                 offset: int,
                 pmf_factory: IPmfFactory):
        super().__init__(pmf_factory)
        self.probabilities = probabilities
        self.size = len(self.probabilities)
        self.offset = offset
        self.__cdf = None
        self.__mean = None
        self.__std = None

    @classmethod
    def from_table(cls,
                   table: Dict[int, float],
                   pmf_factory: IPmfFactory) -> "IntegerInterval":
        min_outcome: int = min(table.keys())
        max_outcome = max(table.keys())
        size = max_outcome - min_outcome + 1
        probabilities = [0.0] * size
        for outcome, probability in table.items():
            probabilities[outcome - min_outcome] = probability
        return cls(probabilities, min_outcome, pmf_factory)

    def cdf(self):
        if self.__cdf is None:
            self.__cdf = []
            prefix_sum = 0
            for probability in self.probabilities:
                self.__cdf.append(prefix_sum)
                prefix_sum += probability
        return self.__cdf

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        return map(lambda i: (self.offset + i, self.probabilities[i]),
                   range(self.size))

    def __add_helper(self, other):
        probabilities = []
        for s in range(self.size):
            probabilities.append(0)
            for i in range(s + 1):
                probabilities[s] += \
                    self.probabilities[i] * other.probabilities[s - i]
        for s in range(self.size, self.size + other.size - 1):
            probabilities.append(0)
            for i in range(s - self.size + 1, min(other.size, s + 1)):
                probabilities[s] += \
                    self.probabilities[s - i] * other.probabilities[i]
        return \
            self._pmf_factory.ints(probabilities, self.offset + other.offset)

    def __add__(self, other):
        if isinstance(other, int):
            return \
                self._pmf_factory.ints(self.probabilities, self.offset + other)
        if self.size <= other.size:
            return self.__add_helper(other)
        else:
            return other.__add_helper(self)

    def __radd__(self, other):
        return self.__add__(other)

    def adv(self):
        probabilities = []
        cdf = self.cdf()
        for i in range(len(self.probabilities)):
            probabilities.append(cdf[i] * self.probabilities[i] +
                                 self.probabilities[i] *
                                 (cdf[i] + self.probabilities[i]))
        return self._pmf_factory.ints(probabilities, self.offset)

    def ge(self, threshold: int):
        cdf = self.cdf()
        probabilities = []
        threshold_index = threshold - self.offset
        for i in range(threshold_index):
            probabilities.append(self.probabilities[i] * cdf[threshold_index])
        for i in range(threshold_index, len(self.probabilities)):
            probabilities.append(
                self.probabilities[i] * (1 + cdf[threshold_index]))
        return self._pmf_factory.ints(probabilities, self.offset)

    def times(self, n, op=operator.__add__):
        return functools.reduce(op, itertools.repeat(self, n))

    def __union_helper(self, other):
        probabilities = self.probabilities + \
                        [0] * max(0, other.offset + other.size
                                  - self.offset - self.size)
        for i in range(other.size):
            probabilities[other.offset - self.offset + i] += \
                other.probabilities[i]
        return self._pmf_factory.ints(probabilities, self.offset)

    def union(self, other):
        if not isinstance(other, IntegerInterval):
            return super(self.__class__, self).union(other)
        if self.offset <= other.offset:
            return self.__union_helper(other)
        else:
            return other.__union_helper(self)

    def scale_probability(self, scale: float):
        return self._pmf_factory.ints([p * scale for p in self.probabilities],
                                      self.offset)

    def p(self, outcome: int) -> float:
        return self.probabilities[outcome - self.offset]

    def outcome_to_str(self, outcome: int) -> str:
        return "{:>3}".format(outcome)

    def stats(self) -> PmfStats:
        return PmfStats(self.mean(),
                        self.std(),
                        self.percentile(0.8),
                        self.percentile(0.2))

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
        return map(lambda i: self.offset + i, range(self.size)),\
            map(lambda i: 1 - cdf[i], range(self.size))
