"""Dataset-specific entry classes (thin subclasses)."""

from .rigor_base import RigorBase


class RIGOR_MATH(RigorBase):
    dataset_mode = "math"


class RIGOR_MEDMCQA(RigorBase):
    dataset_mode = "medmcqa"
