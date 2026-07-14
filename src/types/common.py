import pandas as pd
from typing import Union, Literal

# Domain Primitive Aliases
type Ticker = str
type SpreadId = str
type GroupId = str
type ReturnType = Literal["raw", "demeaned", "cum_raw", "cum_demeaned"]

# Complex Structural Aliases
type GroupedSeries = dict[GroupId, pd.Series]
type GroupedFrames = dict[GroupId, pd.DataFrame]
type UniverseReturns = Union[pd.DataFrame, GroupedFrames]

from typing import Iterable, TypeAlias
