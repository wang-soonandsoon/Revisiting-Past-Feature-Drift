# Source correspondence

This release extracts the feature-level core from the CHR / DAUC-R research
workspace for *Revisiting Past Feature Drift for Exemplar-Free
Class-Incremental Learning*.

The source checkout was at commit
`a600ed302b4080d89af59f32f55b9c604ab01fc2` when this package was prepared on
2026-11-17. The core source files below had no local modifications.

| Published code | Original research source |
| --- | --- |
| `chr_core/cumulative.py` | `dauc_r/round15_geometry.py` |
| `chr_core/current_only.py::select_passive_chains` | `dauc_r/round9_geometry.py` |
| `chr_core/correction.py::corrected_statistics` | `dauc_r/round9_geometry.py` |
| `chr_core/_utils.py` | Four shared helpers in `dauc_r/round9_geometry.py` |
| `chr_core/statistics.py` | `ClassStatistics`, `class_statistics`, `fit_linear_transport`, and `transport_statistics` in `dauc_r/diagnostics.py` |
| `tests/test_cumulative.py` | `tests/test_round15_geometry.py` |
| `tests/test_correction_and_current_only.py` | `CorrectedStatisticsTests` and `PassiveChainTests` in `tests/test_round9_geometry.py` |

Functions were extracted with their arithmetic and existing validation intact.
Imports and module descriptions were adapted to the standalone package.
The helper `_paired_inputs` no longer changes PyTorch's global CPU thread
count. The example and relevant tests set their own thread count explicitly.

The prior-support test, synthetic demonstration, package metadata, and public
documentation were added for this release. The demonstration's moments and
linear propagation use the included double-precision helper path. It does not
reproduce the native image-training, FP32 statistics, sampling-factor, or
classifier-alignment execution of a full benchmark run.

The method correspondence follows the workspace manuscript's
`sections/method.tex`, `sections/algorithm.tex`, and
`sections/joint_transport_supp.tex`. Experimental runners, dataset material,
weights, cached features, manuscript files, and research logs are outside this
release.
