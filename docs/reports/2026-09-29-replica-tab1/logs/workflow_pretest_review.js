export const meta = {
  name: 'uags-pretest-review',
  description: 'Adversarial review of the UAGS Replica evaluation pipeline before the one-shot test run, fixes, and the P3 mixture predictive attempt',
  phases: [
    { title: 'Review', detail: '6 dimension reviewers (read-only)' },
    { title: 'Merge', detail: 'merge duplicate findings' },
    { title: 'Verify', detail: '3 adversarial verifiers per finding' },
    { title: 'Fix', detail: 'one fixer applies confirmed findings + tests' },
    { title: 'P3', detail: 'implement the probability-space (mixture) predictive' },
    { title: 'Critic', detail: 'completeness critic + full CPU test suite' },
  ],
}

const REPO = '/home/thanhnc/gauss_ws/src/uags'
const COMMON = `Project: UAGS, an RA-L paper on a per-Gaussian von Mises-Fisher (vMF) semantic posterior for 3D Gaussian Splatting
(repo ${REPO}; paper papers/main.tex; code usplat/ and scripts/). The Replica Table I pipeline is about to run its TEST split ONCE
(6 scenes), so any bug found after that run would force a second test run. Your job is part of a pre-test review.
Context files:
- ${REPO}/docs/reports/2026-09-29-replica-tab1/README.md (run report: protocol, findings so far, memory fixes)
- ${REPO}/docs/reports/2026-09-29-replica-tab1/logs/eval_val_partial_2013.log (partial validation numbers, office_0 only)
- ${REPO}/docs/reports/2026-09-29-paper-revision-1/logs/fact_sheet.md (method as implemented + honesty rules)
- papers/main.tex: Table I (label tab:replica), the metrics/setup text, Table III (tab:ablation), the views figure text.
Protocol: Replica (Semantic-NeRF renders), map Sequence_1 every 5th frame (180), evaluate Sequence_2 every 10th frame (90) at
downscale 2 (240x320); validation scenes office_0, room_2 fit one temperature per row (P2: beta0, a, b, g); test scenes room_0,
room_1, office_1-4 are evaluated ONCE with the frozen temperatures. Extractor LSeg (ViT-L/16, 512-d tokens at stride 8), PCA-64
fit on 20 mapping frames, 5-template prompt ensemble, queries = classes present in each scene's evaluation frames.
Pipeline: scripts/replica_tab1.sh -> scripts/fit_gs.py, scripts/run_mapping.py, scripts/eval_seg.py (rows: ours_P0/P1/P2,
ours_scalar, baselines from usplat/baselines.py via --row-factory usplat.baselines:row_factory, and the 2D row via --with-2d);
scripts/make_tables.py writes the LaTeX rows. Views figure: scripts/eval_views.py + scripts/replica_views.sh. Table III:
usplat/ablation_rows.py + scripts/replica_tab3.sh. GPU scheduling: scripts/gpu_gate.sh, usplat/gpu_reserve.py.
Recent changes worth special scrutiny (git log -15 --stat shows them): the 2D row and frozen_on in eval_seg.py; the token
pre-extraction restructure of run_mapping.py (compare with git show 60d7796:scripts/run_mapping.py for equivalence); the
estimator "rbar", calib_mode rename ("history" default, "pooled" flat, "max"), and None-for-zero tensors in vmf_field.state_dict;
NormLift 1 - C_inter channel; ablation_rows; eval_views; make_tables; gpu_reserve.
HARD RULES: do NOT use the GPU (another project is running MASt3R on it; run python with CUDA_VISIBLE_DEVICES="" and
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. from ${REPO}); do NOT git commit, push, checkout or reset; do NOT write under docs/reports/;
do NOT modify tracked __pycache__ files. Plain simple English in everything you write.`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          where: { type: 'string', description: 'file:line (one or more)' },
          problem: { type: 'string' },
          failure_scenario: { type: 'string', description: 'concrete input/state -> wrong number, crash, or misleading claim' },
          evidence: { type: 'string', description: 'what you ran or read that shows it (commands + outputs, code lines)' },
          fix: { type: 'string' },
        },
        required: ['title', 'severity', 'where', 'problem', 'failure_scenario', 'evidence', 'fix'],
      },
    },
    checked_ok: { type: 'array', items: { type: 'string' }, description: 'things you checked that are correct (one line each)' },
  },
  required: ['findings', 'checked_ok'],
}

const DIMENSIONS = [
  { key: 'eval-core', prompt: `Review the evaluation core for correctness: scripts/eval_seg.py (load_scene with CPU freezing and
frozen_on buffer sharing, the 2D row: PixelSource, frame_features_2d, token pre-extraction, scene_views_2d, how 2D views are paired
with 3D views, fit_on, evaluate_scene, betas_used, check_fit_settings, text_embeddings and the text cache keys, map_labels and
queries="present") and usplat/seg_eval.py (render_row, class_scores, evaluate_view, coverage rule, tie rule, BetaFitter, summarize,
ranking_signals, AUSE/AURC, ECE/NLL/Brier). Check that every metric matches its definition in papers/main.tex (tab:replica caption,
metrics paragraph). Check the 2D row is scored on exactly the same labelled pixels, text and label grid as the 3D rows. Check that
freezing on the CPU then moving to the GPU cannot change a number (dtype, device of text vs buffer, shared tensors mutated later?).
Run the existing tests (tests/test_seg_eval.py, tests/test_metrics.py) and write small CPU experiments where useful.` },
  { key: 'mapping', prompt: `Review the mapping pass: scripts/run_mapping.py after the token pre-extraction restructure (commits
5b21781 and cc65f08) against the version at git 60d7796 (git show 60d7796:scripts/run_mapping.py). Is the set and order of keyframes,
the PCA fit, the centre, the feature z per frame, the CSS vocabulary/text, the field update and the saved checkpoint identical in
content to the old code (apart from timing fields)? Are the timing fields (ms_per_keyframe_field, _update, _extract, peak_alloc_*)
measuring what their names and comments say, suitable for the paper's run-time paragraph? Check usplat/gpu_reserve.py (set fraction,
reserve, margin, the small-block pool, interaction with set_per_process_memory_fraction and later allocations; any way it can make a
job fail that would otherwise pass) and scripts/gpu_gate.sh (USGLoc detection by /proc cwd, lock, re-check, exit codes, poll).
Also check scripts/fit_gs.py is called by replica_tab1.sh with the same settings used for the office_0 map (max_res, seed, iters).
You may use the mock backend (run_mock) and CPU tests; do not run the gsplat backend.` },
  { key: 'baselines', prompt: `Review the baselines for correctness and FAIRNESS: usplat/baselines.py (Averaging = GradBP, NormLift
incl. the new "normlift_1-Cinter" channel, ViewCount, CSS with Yager conflict, default_baselines, baselines_state/load_baselines,
row_factory) and how scripts/run_mapping.py feeds CSS (css_vocabulary uses the classes PRESENT in the scene's evaluation frames;
note the baselines.py docstring (d) says CSS receives the frozen evaluation vocabulary and never the classes present in a test
scene: resolve which is true and whether the paper text (tab:replica, setup text) describes it correctly). Is any baseline given an
advantage or a handicap the paper does not state (e.g. CSS vocabulary = the eval-present classes; CSS temperature on log q; 2D row
covering every pixel while 3D rows lose uncovered pixels; NormLift/averaging/viewcount sharing one predictive; view count's
definition)? The partial validation numbers show CSS best calibrated (ECE 4.7 vs ours 7.1): is that a real property or an artefact
(e.g. of the CSS vocabulary, of the tie rule, of probs rows vs feature rows under the same temperature fit)? Check the NormLift
equations against arXiv 2609.18898 descriptions in the code docstring and papers/main.tex. Run tests/test_baselines.py.` },
  { key: 'views-ablations', prompt: `Review the views figure and Table III code: scripts/eval_views.py, scripts/replica_views.sh,
usplat/seg_eval.py views_cross_ause/aligned_error_targets/error_targets/cross_ause, usplat/ablation_rows.py, scripts/replica_tab3.sh,
and the new FieldConfig options in usplat/vmf_field.py (estimator "rbar", calib_mode "history"/"pooled"/"max" and the from_state
rename shim, state_dict None for all-zero tensors and from_state restoring zeros; tensors that must NOT be saved as None).
Check against papers/main.tex (Table III rows and the ablation paragraph, the views figure text: AUSE of u_epi vs e_red and u_ale vs
e_irr averaged over the 10-, 30-, 90-frame fields). Specific questions: do the k-view fields share the full field's PCA (same basis)
so text and features are comparable? --max-keyframes plus the pose-gap keyframe filter: is k really k? Does the Tab III "Pixels
instead of views" variant use evidence="pixels", epistemic="pixels" and change_test False as the paper says? Is est_legacy the
paper's r_tilde? Are temperatures refit for every ablation row on val? Does replica_views.sh delete fields only after their JSON
is written? Run tests/test_ablation_rows.py, tests/test_seg_eval.py -k views, tests/test_state.py, tests/test_change.py.` },
  { key: 'pipeline-tables', prompt: `Review the pipeline scripts and table generation for reproducibility and the test-once rule:
scripts/replica_tab1.sh (STAGE val/test, reuse of existing maps/fields, logs overwritten on rerun: eval_val.log was overwritten
once and a partial result was lost; exit handling under set -euo pipefail; the test stage refusing to run without eval_val.json),
configs/replica_val.json, configs/replica_test.json, configs/vocab_replica.json (templates_by_extractor lseg), scripts/make_tables.py
(row order, columns, units: mIoU/Acc/ECE/AURC/AUSE in %, Brier x100, NLL nats; AUSE_u signals: NormLift 1-R and 1-C_inter, CSS
css_var_top, ours u_ale; the "same predictive as averaging" multicolumn; N and std) against the tabular in papers/main.tex
(tab:replica). Is there anything that could let test data leak into a choice (temperatures, templates, extractor, PCA, tau_cov,
undef_frac)? Are all random seeds fixed? Is the 5-template ensemble what the paper says? Check disk usage assumptions (disk is 98%
full; fields are ~0.8-1.3 GB). Propose concrete fixes (e.g. timestamped or appended logs).` },
  { key: 'field-math', prompt: `Review the field read-outs and predictives used in Table I against papers/main.tex (method section:
view weights, support, epistemic u_epi = 1/(1+W1), Kish agreement rho_hat, aleatoric u_ale = 1 - rho_hat with the tau_min gate,
joint-prior kappa_hat, P0 = mu*, P1 = A_d(kappa_post) mu*, P2 adaptive beta = beta0 exp(-a u_epi - b u_ale - g 1[undef])):
usplat/vmf_field.py, usplat/vmf_math.py, usplat/predictive.py (gaussian_buffer, adaptive_beta, fit_temperature, fit_adaptive),
usplat/rows.py (field rows, channels u_epi, u_ale, spread). Look for numerical problems at the real scale: 1M Gaussians, most with
support < 1 view, u_ale defined for 12-22% of Gaussians but ~82% of pixel mass, u_ale values tiny (mean 0.007-0.02). Is P2's fit
(b = 11.8 fitted on val) well posed; can the adaptive beta blow up or go negative; is the "undefined" indicator rendered right?
Does freezing a FieldRow snapshot exactly what the paper defines? Run tests/test_estimators.py tests/test_legacy.py and write small
CPU checks (e.g. a synthetic field with the observed support distribution).` },
]

phase('Review')
const reviews = await parallel(DIMENSIONS.map(d => () =>
  agent(`${COMMON}\n\nYOUR DIMENSION: ${d.key}\n${d.prompt}\n\nReport only real defects with concrete evidence (commands you ran and their output, code lines). Severity: critical = a wrong or misleading number in the paper's tables or a crash of the test run; major = a wrong claim, unfair comparison, or reproducibility gap; minor = clarity/robustness. List what you checked and found correct in checked_ok.`,
    { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS })))
const all = []
reviews.forEach((r, i) => { if (r) r.findings.forEach(f => all.push({ ...f, dim: DIMENSIONS[i].key })) })
const checkedOk = reviews.filter(Boolean).flatMap((r, i) => r.checked_ok.map(c => `[${DIMENSIONS[i].key}] ${c}`))
log(`${all.length} raw findings from ${reviews.filter(Boolean).length} reviewers`)

phase('Merge')
const MERGED = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' }, title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          where: { type: 'string' }, problem: { type: 'string' }, failure_scenario: { type: 'string' },
          evidence: { type: 'string' }, fix: { type: 'string' }, sources: { type: 'array', items: { type: 'string' } },
        },
        required: ['id', 'title', 'severity', 'where', 'problem', 'failure_scenario', 'evidence', 'fix', 'sources'],
      },
    },
  },
  required: ['findings'],
}
const merged = all.length === 0 ? { findings: [] } : await agent(
  `Merge duplicate review findings. Below is a JSON list of findings from independent reviewers of the same code base. Two findings are
duplicates only if they describe the same defect (same root cause). Merge duplicates into one entry (keep the highest severity, the
union of evidence and locations, the best fix; list the reviewer dims in sources). Do NOT drop, judge or soften any finding that is
not a duplicate; do not add new ones. Give each merged finding an id F01, F02, ... ordered by severity.\n\n${JSON.stringify(all)}`,
  { label: 'merge', phase: 'Merge', schema: MERGED, effort: 'medium' })
log(`${merged.findings.length} findings after merging duplicates`)

phase('Verify')
const VERDICT = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'true if the defect is real as stated (or real with a corrected description)' },
    severity: { type: 'string', enum: ['critical', 'major', 'minor', 'none'] },
    reasoning: { type: 'string' },
    evidence: { type: 'string', description: 'what you ran/read to decide' },
    corrected_fix: { type: 'string', description: 'a better or safer fix if the proposed one is wrong or incomplete; else empty' },
  },
  required: ['real', 'severity', 'reasoning', 'evidence', 'corrected_fix'],
}
const LENSES = [
  'Try to REFUTE it: find the code path, test or argument that shows the defect does not exist or cannot happen with the real data and settings. If you cannot refute it with evidence, say real=true.',
  'Check it by EXECUTION: write and run a minimal CPU reproduction (CUDA_VISIBLE_DEVICES=""), or run the relevant tests; decide from what actually happens.',
  'Check the IMPACT and the FIX: would it change a number or claim in the paper or crash the test run with the real settings (Replica, 620k-954k Gaussians, LSeg, 90 eval frames)? Is the proposed fix correct and minimal, or would it break something else?',
]
const verified = await parallel(merged.findings.map(f => () =>
  parallel(LENSES.map((lens, j) => () =>
    agent(`${COMMON}\n\nA reviewer reported this defect:\n${JSON.stringify(f, null, 1)}\n\nYOUR TASK: ${lens}\nBe concrete; do not modify any repository file (you may write scratch files under /tmp/claude-1005 only).`,
      { label: `verify:${f.id}:${j}`, phase: 'Verify', schema: VERDICT })))
    .then(vs => {
      const ok = vs.filter(Boolean)
      const yes = ok.filter(v => v.real && v.severity !== 'none')
      return { ...f, votes: ok.map(v => ({ real: v.real, severity: v.severity, reasoning: v.reasoning, corrected_fix: v.corrected_fix })), confirmed: yes.length >= 2 }
    })))
const confirmed = verified.filter(Boolean).filter(f => f.confirmed)
const rejected = verified.filter(Boolean).filter(f => !f.confirmed)
log(`${confirmed.length} confirmed, ${rejected.length} rejected`)

phase('Fix')
const FIXREPORT = {
  type: 'object',
  properties: {
    per_finding: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, outcome: { type: 'string', enum: ['fixed', 'skipped', 'no_change_needed'] },
      what: { type: 'string' }, files: { type: 'array', items: { type: 'string' } } },
      required: ['id', 'outcome', 'what', 'files'] } },
    tests_command: { type: 'string' }, tests_output_tail: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['per_finding', 'tests_command', 'tests_output_tail', 'notes'],
}
const fixReport = confirmed.length === 0 ? null : await agent(
  `${COMMON}\n\nYou are the single FIXER. Apply the fixes for these CONFIRMED findings (each has the reviewer's fix and the verifiers'
votes, some with a corrected fix: prefer the corrected fix when verifiers agree it is better). Keep each change minimal and in the
style of the surrounding code (docstrings, comment density). Add or update a CPU test for every behaviour you change. Do not change
an evaluation definition just to make numbers look better; if a finding requires a paper-text change, edit papers/main.tex and keep
it compiling (tectonic at /home/thanhnc/gauss_ws/third_party/tectonic/tectonic; build in a scratch copy of papers/, never commit
build products). If a finding needs a decision from the lead (e.g. changing the protocol), mark it skipped and explain.
Finally run: CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/
and report the tail. Restore any tracked __pycache__ file you changed (git checkout -- usplat/__pycache__).\n\nCONFIRMED FINDINGS:\n${JSON.stringify(confirmed, null, 1)}`,
  { label: 'fix', phase: 'Fix', schema: FIXREPORT })

phase('P3')
const P3 = await agent(`${COMMON}\n\nIMPLEMENT an additional predictive row "ours_P3" (a method attempt; the validation split will decide
whether it stays). Motivation: in the partial validation run CSS (a closed-set Dirichlet that renders per-Gaussian class
PROBABILITIES) had ECE 4.7 vs 7.1 for our best feature-space predictive (P2), at equal mIoU. Our P0/P1/P2 render the per-Gaussian
FEATURE (mu* or A_d(kappa) mu*) and then apply a softmax per pixel. P3 renders, per Gaussian, the class predictive of the
queried classes and mixes those distributions with the splat weights (a mixture over the Gaussians on the ray, as CSS does), so it
stays OPEN-VOCABULARY: the per-Gaussian probabilities are computed at query time from mu* (or the posterior) and the query texts,
nothing is fixed at mapping time.
Design (keep it simple and principled; justify choices in the docstring):
- per Gaussian n and queried class c: q_n(c) = softmax_c(beta_g * m_n . t_c), with m_n = A_d(kappa_post,n) mu*_n (the posterior
  mean direction, so weakly supported Gaussians give flatter distributions) and t_c the projected text (same as the other rows);
  an unseen Gaussian (no evidence) gets the uniform distribution;
- beta_g (the per-Gaussian temperature) is a hyperparameter: it must be chosen on the VALIDATION scenes only. Implement it so the
  evaluation can produce a small grid of rows (e.g. ours_P3_b10, ours_P3_b20, ours_P3_b40) or a single row with a given value;
- the row kind is "probs" (usplat/rows.py, usplat/seg_eval.py already support probs rows: q_p = sum_n w q_n / alpha, then a
  post-hoc temperature on log q_p fit on val); it carries the same uncertainty channels u_epi and u_ale as ours_P0;
- rows need the scene's projected text at build time: thread the text (K, d) from scripts/eval_seg.py load_scene into the row
  construction without breaking the row-factory interface or eval_views.py; memory: the buffer is (N, K) with K ~ 20-30 classes.
Add CPU tests (a toy field where P3 must equal the analytic mixture; uniform for unseen Gaussians; the probs path through
evaluate_view; the text threading in eval_seg with the mock backend). Update scripts/make_tables.py only if the paper table should
list P3 (do NOT edit papers/main.tex: the lead decides after validation). Update docstrings. Run the full CPU test suite at the end
(CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/) and report the tail.
Return: files changed, design choices, how to enable P3 in scripts/replica_tab1.sh (exact flag), test output.`,
  { label: 'p3', phase: 'P3' })

phase('Critic')
const critic = await agent(`${COMMON}\n\nYou are the COMPLETENESS CRITIC after a review/fix round and a new predictive (P3). Inspect the
uncommitted diff (git diff; git status) and answer: (1) Does any change alter a number the validation or test run would produce
in a way that was not intended? (2) Is every confirmed finding below really fixed (read the code, run the tests)? (3) What in the
test-run path is still unverified or risky (list concretely)? (4) Run the FULL CPU test suite:
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -q -p no:cacheprovider tests/  and report the tail.
(5) Check papers/main.tex still compiles if it was changed (tectonic in a scratch copy). Do not edit files except to fix an obvious
regression you find (say so).\n\nConfirmed findings and the fixer's report:\n${JSON.stringify({ confirmed: confirmed.map(f => ({ id: f.id, title: f.title, fix: f.fix })), fixReport }, null, 1)}\n\nP3 implementer's report:\n${P3}`,
  { label: 'critic', phase: 'Critic' })

return { raw_count: all.length, merged: merged.findings.length, confirmed, rejected, checkedOk, fixReport, P3, critic }
