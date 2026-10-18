# ASHG poster

`simulation.py` collects the MSI `grid2` results, `figures.py` draws the panels, and `build_pptx.py` fills the lab's 48 × 36 inch template. The older HTML/PDF builder is separate from this PowerPoint workflow.

Use `uv` from the repository root. MSI access must already be connected for the fetch:

```sh
uv run python docs/poster/ashg2026/simulation.py fetch
uv run python docs/poster/ashg2026/figures.py
uv run --no-project --with defusedxml --with lxml --with pillow python docs/poster/ashg2026/build_pptx.py
```

The template defaults to `~/Downloads/ASHG draft.PH.pptx`, the output to `~/Downloads/ASHG SV-PGS poster.pptx`, and the untracked author information comes from `authors.local`. Positional arguments override the template, output, summary and figure directory. Keep the template, author file and generated artifacts out of git. Render the resulting PowerPoint to PDF and visually inspect it after rebuilding; updating the PowerPoint does not update an existing PDF.

Missing results produce explicit placeholders. A previous optional architecture panel is removed when the new summary cannot supply it. Never use layout-test values as scientific results.

When only part of the simulation grid has completed, the narrative reports the actual paired-comparison and replicate counts and labels the results as interim. Cells without enough replicates do not retain template numbers. Scheduler-killed attempts may still say `running` until reconciled against job history; an active writer's lock must never be bypassed during such a repair.

## Reading the summary

- `coverage` counts completed, failed and running arm attempts, including cells without any paired result. It does not count scheduler jobs that have not started or prove that a recorded running attempt is still alive.
- Means and 95% Student t intervals condition on completed replicates; an interval's `replicates` is its own denominator. With fewer than two finite values, its interval is unavailable. Narrative comparisons require at least three replicates for the relevant comparison.
- Power is the fraction of valid one-sided paired DeLong comparisons favouring the SV arm, with a Wilson 95% interval. An undefined comparison is excluded from this denominator, but its finite AUC estimate can still enter the mean estimate. Failed fits are reported separately and must not be interpreted as negative results.
- No causal SV is a biological control, not necessarily an equal-accuracy null: LD tagging and regularization can still change prediction. An interval spanning zero does not establish equivalence.
- LDpred2's SNV subset has HapMap3 density; it is not the actual HapMap3 variant list. Its all-class comparator includes measured variants with MAF ≥ 1%. SV-PGS retains rarer variants as well, so this is not a comparison with identical variant inputs.
- SV effect readouts enter the summary only for completed SV-arm attempts. Their correlations and length-bin means are descriptive; variants in LD are not independent replicates.

The All of Us panels remain placeholders until authorized, in-perimeter study results are available. This simulation continuation does not launch that study.
