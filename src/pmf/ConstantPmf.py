from typing import *
from .Pmf import IPmfFactory
from .TablePmf import TablePmf

TOutcome = TypeVar("TOutcome")


class ConstantPmf(TablePmf[TOutcome]):
    def __init__(self, constant: TOutcome, pmf_factory: IPmfFactory):
        super().__init__({constant: 1}, pmf_factory)
