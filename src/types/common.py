from typing import Literal

import pandas as pd

# Domain Primitive Aliases
type Ticker = str
type SpreadId = str
type GroupId = str
type ReturnType = Literal["raw", "demeaned", "cum_raw", "cum_demeaned"]

# Complex Structural Aliases
type GroupedSeries = dict[GroupId, pd.Series]
type GroupedFrames = dict[GroupId, pd.DataFrame]
type UniverseReturns = pd.DataFrame | GroupedFrames
