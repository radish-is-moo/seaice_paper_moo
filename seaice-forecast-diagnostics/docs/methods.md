# Scientific contract

- Experiment: 2023–2025 ver2; training 1979–2019; validation 2020–2022. Three models: CNN, U-Net, grid GraphSAGE GNN.
- Grid: 101 × 1440, 0.25°, 65–90°N. Eight predictors in order: t2m, d2m, u10, v10, msl, sst, sic, sit. History: 15 days; direct forecast: 30 days.
- Evaluation: 94 initializations per year, 282 total; every target day in July–October. Leads 1–30, with early 1–5, middle 6–15, late 16–30. Growth is late minus early daily RMSE within each model and initialization.
- Static ocean mask: 101206 cells; combined NSR union: 29595 cells. Open water is included. Regional rectangles overlap at some inclusive boundaries; combined NSR is a union, not a sum of rectangle areas.
- Reference: PIOMAS SIC; it is not an independent observational truth field. Saved reference values are preserved by the calculation.
- RMSE: sqrt(sum(area × squared error) / sum(area)) over valid cells for each model–initialization–lead–category combination. Spatial r is the area-weighted spatial pattern correlation, not anomaly correlation. Undefined categories/correlations remain missing.
- Target-date classes: open water SIC < 0.15; MIZ 0.15 ≤ SIC ≤ 0.80; compact ice SIC > 0.80.
- Initial-condition classes remain fixed throughout all leads. Gradient thresholds are ocean area-weighted 33rd/67th/90th percentiles for each initialization over the pan-Arctic ocean. Initial thickness uses open water, <0.5 m, 0.5–<1.5 m, ≥1.5 m. Wind combinations use initially ice-covered cells, thin versus medium/thick ice, and wind ≤ or > the pan-Arctic 67th percentile.
- Non-MIZ pools open-water and compact-ice SSE and area before taking the square root. Figure 7 compares MIZ and non-MIZ on common valid model–initialization–lead support.
- The three-model mean is an equal mean of separately calculated metrics, never a mean forecast field.
- Transition exposure compares reference state at initialization and target time, using all valid ocean grid–date area as denominator. Six directional pathways sum to total exposure. Intermediate transitions and timing are not recovered.
- Bootstrap: 5000 hierarchical year/within-year non-circular moving-block samples; 7-day primary and 14/30-day sensitivity, seed 20260816. Shared samples preserve paired contrasts. State growth uses three pairwise comparisons; initial-condition growth uses six contrasts per family with corresponding Bonferroni intervals.
- Persistence: calendar-day anomalies relative to 1979–2019; July–October 2023–2025, within-year lags 1–30. First interpolated crossing of r = 0.5. Maps require ≥60 pairs at every lag and anomaly SD ≥0.02; uncrossed values are marked >30 days. MIZ frequency uses its own valid-ocean support.
- Figure 3: lead-30 case chosen by eligible model-mean MIZ-minus-compact RMSE contrast, with support and model-agreement filters; ice-edge contours appear only in NSR zoom panels.
- Training loss retains the original finite-target, unweighted formulation. Ocean/area weighting is applied during manuscript verification, not retroactively imposed on training.

These are diagnostic associations. Wind bins and endpoint transitions do not establish causal error amplification or validate Table 3's physical hypotheses.
