from typing import TypeVar, List, Dict, Tuple, Any
from .Pmf import Pmf, IPmfFactory
from .IntegerInterval import IntegerInterval
from .TablePmf import TablePmf
from .ConstantPmf import ConstantPmf
from .JointPmf import JointPmf

TOutcome = TypeVar("TOutcome")


class PmfFactory(IPmfFactory):
    def from_object(self, obj: Any) -> Pmf[Any]:
        if isinstance(obj, Pmf):
            return obj
        elif isinstance(obj, int):
            return self.ints([1], obj)
        else:
            return self.const(obj)

    def from_table(self, table: Dict[TOutcome, float]) -> Pmf[TOutcome]:
        outcome = next(iter(table.keys()))
        if isinstance(outcome, int) and not isinstance(outcome, bool):
            return IntegerInterval.from_table(table, self)
        else:
            return self.table(table)

    def joint(self, pmfs: List[Pmf[TOutcome]]) -> Pmf[Tuple]:
        return JointPmf(pmfs, self)

    def ints(self, probabilities: List[float], offset: int = 0) -> Pmf[int]:
        return IntegerInterval(probabilities, offset, self)

    def table(self, table: Dict[TOutcome, float]) -> Pmf[TOutcome]:
        return TablePmf(table, self)

    def const(self, constant: TOutcome) -> Pmf[TOutcome]:
        return ConstantPmf(constant, self)

    def if_(self,
            condition_pmf: Pmf[bool],
            then_pmf: Any,
            else_pmf: Any) -> Pmf[Any]:
        scaled_then_pmf = self.from_object(then_pmf)\
            .scale_probability(condition_pmf.p(True))
        scaled_else_pmf = self.from_object(else_pmf)\
            .scale_probability(condition_pmf.p(False))
        return scaled_then_pmf.union(scaled_else_pmf)
