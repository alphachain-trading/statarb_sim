# statarb_sim

A systematic statistical arbitrage (pairs, sparse+dense portfolio spreads) trading research platform for S&P 500 equities. Built as a rigorous research tool, not a black-box strategy — the codebase is designed for transparency, reproducibility, and extension.

## What this is

A complete pipeline for pairs trading research:
- **Data**: S&P 500 sector universe download and caching via yfinance
- **Residualization**: Causal two-stage OLS residualization (Fama-French inspired), removing market and sector factor exposure
- **Candidate generation**: Weekly walk-forward pair candidate panels with PCA-based hedge ratios and mean-reversion diagnostics
- **Simulation**: Event-driven pairs trading simulator with EWM z-score signals, position sizing, transaction costs, and borrow costs
- **Research tools**: Z-score spectrum explorer, backward/forward gain analysis, entry feature validation pipeline *(coming)*

## What we found

The system trades equity pair spreads within S&P 500 sectors. Across 10 sectors, the strategy generates consistent positive edge (profit factors 1.0–2.1 depending on sector) with win rates of 55–73%. Edge is concentrated in mean-reversion of causal residual spreads rather than raw price spreads.

**Validated findings:**
- `x_area_asymmetry_ewm` is a statistically significant entry quality signal (z-separation 1.83, half-split stable)
- hl252/hl378 residual timescales structurally outperform hl63 (62–72% vs 54–57% win rate)
- Risk-free rate inclusion in residualization is material in trending rate environments

**Negative results (documented):**
- Blended z-score entry modes show no improvement
- Cross-timescale signal gates neutral to harmful
- Backward-window gain does not predict forward-window gain as an entry criterion
- Feature-weighted position sizing degrades Sharpe at every tested multiplier

## Quickstart

```bash
git clone https://github.com/alphachain-trading/statarb_sim.git
cd statarb_sim
python -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

## Architecture

```
run_me.py                  # staged pipeline runner
config/                    # universe definitions, run configs
src/
    data/                  # universe download and market data
    residuals/             # causal residualization
    candidates/            # pair candidate panel creation
    simulator/             # event-driven backtest engine
    analytics/             # spread primitives and feature computation
fixtures/materials_v1/     # pre-built materials sector panel for demo
```

Each stage is idempotent — re-running skips completed work unless `--force` is passed.

## Status

| Component                          | Status |
|------------------------------------|---|
| Data pipeline                      | ✅ complete |
| Residualization                    | ✅ complete |
| Pair candidate panel               | ✅ complete |
| Simulator                          | ✅ complete |
| Research tools (z-spectrum, bw/fw) | 🔄 migration in progress |
| Teaching notebooks                 | 🔄 planned |
| Multi-timescale sweep runner       | 🔄 planned |
| Methodology, validated results     | 🔄 planned |


## License

MIT

## Author

Nikolaj Nock
[LinkedIn](https://www.linkedin.com/in/nikolaj-karl-nock)


