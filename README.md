# Multi-target Protein Binder Design with Multistate RFdiffusion Guidance

This repository includes a concentrated multistate binder modification for extending **single binder backbone / pose** to geometrically compatible **multiple pre-aligned homologous targets** during RFdiffusion inference.

What it does:
- Runs standard RFdiffusion binder generation against one reference target.
- Loads multiple pre-aligned target PDBs plus per-target hotspot residue lists.
- At each denoising step, scores the current binder pose against every target using three simple geometric proxy terms: hotspot proximity loss, clash penalty, and contact reward.
- Aggregates per-target interface losses with a **smooth worst-case** objective by default (`multistate.aggregate_mode=softmax_max`), with `max` available as an ablation.
- Adds a pose-consistency term by default (`multistate.use_pose_consistency=true`) based on binder centroid / principal-axis consistency relative to each target hotspot region.
- Applies a **small binder-only coordinate-space update** with a schedule (`multistate.guide_schedule`) plus gradient clipping and optional bad-step rejection.

What it assumes:
- The targets are already structurally aligned into a common reference frame outside RFdiffusion.
- The intended epitope / hotspot regions are homologous enough that hotspot centroids and local hotspot axes should superpose reasonably well.
- One of the targets can be used as the single RFdiffusion conditioning target. Internally, `multistate.reference_target_idx` is used to set `inference.input_pdb`, and its hotspot list is used as `ppi.hotspot_res`.

Alignment sanity check:
- RFdiffusion does **not** run TM-align, CEalign, or any other structural alignment engine internally for this feature.
- Instead, `multistate.alignment_check=true` compares each target hotspot centroid to the reference hotspot centroid and, when possible, compares the local hotspot principal direction.
- If these checks exceed `multistate.centroid_tolerance` or `multistate.axis_tolerance_deg`, inference aborts so the user can fix the external alignment first.

Important interpretation:
- The outputs should be treated as **multi-target-compatible backbone / pose hypotheses**.
- They are **not** final validated binders.
- This feature does **not** retrain RFdiffusion, change model weights, modify checkpoints, run physics-based binding energy evaluation, or guarantee sequence-level chemical compatibility across homologs.

Modified files for this feature:
- `rfdiffusion/multistate_inference.py`
- `config/inference/multistate_binder.yaml`
- `scripts/run_inference.py`
- `README.md`

New config:
- Use `--config-name multistate_binder` to access the new `multistate.*` options without changing the default single-target config path.
- The main new fields are:
  - `multistate.target_pdbs`
  - `multistate.hotspot_res_by_target`
  - `multistate.reference_target_idx`
  - `multistate.aggregate_mode`
  - `multistate.softmax_temperature`
  - `multistate.guide_scale`
  - `multistate.guide_schedule`
  - `multistate.use_pose_consistency`
  - `multistate.pose_weight`
  - `multistate.hotspot_weight`
  - `multistate.clash_weight`
  - `multistate.contact_weight`
  - `multistate.contact_cutoff`
  - `multistate.clash_cutoff`
  - `multistate.alignment_check`
  - `multistate.centroid_tolerance`
  - `multistate.axis_tolerance_deg`
  - `multistate.grad_clip_norm`
  - `multistate.reject_bad_guidance_step`

Conceptually, the algorithm is:
1. Choose one target as the RFdiffusion conditioning target (`reference_target_idx`).
2. Run the standard RFdiffusion sampling step.
3. Extract the current binder coordinates.
4. Score that binder pose independently against every aligned homolog.
5. Aggregate the per-target interface losses with a smooth worst-case objective.
6. Optionally add pose-consistency loss across targets.
7. Apply a conservative binder-only guidance update.
8. Continue denoising.

Example command:

```bash
python scripts/run_inference.py \
  --config-name multistate_binder \
  inference.output_prefix=outputs/ms_fbar \
  inference.num_designs=10 \
  'contigmap.contigs=[70-70/0 A1-240]' \
  multistate.enable=true \
  'multistate.target_pdbs=[inputs/FNBP1_bar_dimer.pdb,inputs/FNBP1L_bar_dimer.pdb,inputs/TRIP10_bar_dimer.pdb]' \
  'multistate.hotspot_res_by_target=[[A45,A49,A82,A86],[A47,A51,A84,A88],[A46,A50,A83,A87]]' \
  multistate.aggregate_mode=softmax_max \
  multistate.guide_scale=0.05 \
  multistate.use_pose_consistency=true
```

Notes on the example:
- The target PDBs should already be aligned before running RFdiffusion.
- `reference_target_idx=0` by default, so the first target in `multistate.target_pdbs` becomes the RFdiffusion conditioning target unless changed.
- Default single-target behavior is unchanged when `multistate.enable=false` or when the `multistate` block is absent.