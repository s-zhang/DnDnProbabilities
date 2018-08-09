from typing import *
from .Pmf import Pmf, IPmfFactory

TOutcome = TypeVar("TOutcome")


class TablePmf(Pmf[TOutcome]):
    def __init__(self, table: Dict[TOutcome, float], pmf_factory: IPmfFactory):
        super().__init__(pmf_factory)
        self.table = table

    def __iter__(self) -> Iterator[Tuple[TOutcome, float]]:
        return iter(self.table.items())

    def p(self, outcome: TOutcome) -> float:
        return self.table[outcome]

    def scale_probability(self, scale: float):
        return TablePmf({outcome: probability * scale for outcome, probability in self.table.items()}, self._pmf_factory)
