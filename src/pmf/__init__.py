from .PmfFactory import PmfFactory

factory = PmfFactory()
ints = factory.ints
joint = factory.joint
table = factory.table
to_pmf = factory.from_object
if_ = factory.if_

__all__ = [
    "PmfFactory",
    "factory",
    "ints",
    "joint",
    "table",
    "to_pmf",
    "if_"
]
