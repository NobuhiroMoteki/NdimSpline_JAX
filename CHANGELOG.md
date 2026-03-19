# Changelog

## v1.0.1

### Added
- `ndim_spline_jax` package with JAX-native implementation:
  - `compute_coefs`: Tensor-product TDMA solver for N-dim coefficient computation — O(N·M<sup>N+1</sup>)
  - `make_interpolant`: Localized evaluation via `dynamic_slice` — only 4<sup>N</sup> coefficients per query
- `docs/theory_note.tex` / `docs/theory_note.pdf`: LaTeX technical note with full mathematical derivation
- `technical_note_theory.md`: Markdown version of the technical note
- `benchmark.py`: Reproducible time + memory benchmark script
- `pyproject.toml`: Package metadata
- `tests/`: 42 pytest tests covering correctness, gradients, JIT compatibility, and boundary evaluation

### Changed
- `caller.py` and `caller.ipynb` rewritten to use new `ndim_spline_jax` API
- `jupyter_notebooks/`: All 5 dimension-specific notebooks rewritten with new API
- `README.md`: Updated usage section, added performance comparison table and link to technical note
- No dimension limit (previously hardcoded to N ≤ 5)

### Fixed
- 5D coefficient computation bug in legacy code (missing index in loop)

### Removed
- `caller_new.py`: Replaced by `benchmark.py`

### Performance (5D, n=10, CPU, float64)
| Operation | v0.1.2 | v1.0.1 | Speedup |
|---|---|---|---|
| Coefficient computation | 96.6 s | 0.61 s | 158x |
| JIT eval | 3.1 ms | 106 us | 30x |
| JIT grad | 113.5 ms | 111 us | 1,022x |
| JIT value_and_grad | 113.4 ms | 108 us | 1,046x |

### Legacy
- Old modules (`SplineCoefs_from_GriddedData.py`, `SplineInterpolant.py`) moved to `legacy/`

## v0.1.2

Last release before refactoring. Original NumPy/SciPy implementation.

### Known issues
- 5D coefficient computation bug (missing index)
- Dimension limit: N ≤ 5 (hardcoded per-dimension methods)

## v0.3.4

### Added or Changed
- Project title changed.
- Minor changes in README file

## v0.3.3

### Added or Changed
- Typo correction in README file

## v0.3.2

### Added or Changed
- Minor changes in README file

## v0.3.1

### Added or Changed
- Minor change in README file

## v0.3.0

### Added or Changed
- supports of 1,2,3,4,5 dimensions

## v0.2.2

### Added or Changed
- supports of 3 and 4 dimensions
