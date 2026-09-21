# Reproducibility and provenance

## Three levels

1. **Figure reproduction:** included numerical inputs → latest Figures 1–7. No raw climate data or checkpoint is required.
2. **Frozen analysis reproduction:** original 1128 hashed archives + daily cache → regional, state and initial-condition diagnostics, uncertainty estimates and persistence. This can take substantial memory, storage and runtime.
3. **Training:** prepared input data → original architecture/training notebooks → new checkpoints and new inference. Identical weights are not promised: the original CNN/U-Net notebooks do not enforce a complete deterministic training setup, and CUDA algorithms/hardware also affect reproducibility.

The numerical helper functions were carried over from the source project. File paths/imports were made portable; editor-only checks against Word/PDF documents were removed. Notebook outputs and obsolete evaluation cells were removed. Trained model class definitions and hyperparameters were preserved. `source_mapping.json` records original relative file names and SHA-256 identities.

The figure command defaults to bundled frozen inputs. It does not automatically redraw from arbitrary recalculated results. This protects against inadvertently presenting a new experiment as the published figures. The plotting modules expose their input paths through `plot_config.py`; `SEAICE_PLOT_INPUT` can select an equivalent input tree, but Figure 3 needs the matching selected-case archive and Figure 5 needs the band cross-check file. The full evaluator does not automatically assemble that alternative tree.

No claim of a fresh full training or 282-case recomputation is made for this packaging task. See validation.md for the actual tests performed. Source research folders and manuscript files were not modified.

## Figure/code mapping

| Figure | Release module | Numerical source |
| --- | --- | --- |
| 1 | plot_overview / geography | fixed domain boxes |
| 2 | plot_overview | original workflow structure |
| 3 | plot_maps | selected lead-30 frozen case; zoom-only ice edges |
| 4 | plot_regional | ocean regional daily RMSE/r + bootstrap intervals |
| 5 | plot_states | pan-Arctic daily target states / fixed initial conditions |
| 6 | plot_maps | reference anomaly persistence and MIZ frequency |
| 7 | plot_transitions | endpoint exposure, matched absolute state errors, six pathways |

Legacy aggregate filenames beginning with `figure5_condition_` inside the ocean stage refer to NSR intermediate/audit tables; the current Figure 5 explicitly reads the separate pan-Arctic state and condition summaries.
