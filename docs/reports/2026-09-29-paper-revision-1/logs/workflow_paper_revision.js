export const meta = {
  name: 'uags-paper-revision-1',
  description: 'Rewrite the UAGS paper to the implemented method: parallel section drafts, integration in ieeeconf, 3-lens review, fixes, compile',
  phases: [
    { title: 'Draft', detail: 'front matter, related work, method, experiments, figures, template' },
    { title: 'Integrate', detail: 'assemble main.tex, park removed sections, compile' },
    { title: 'Review', detail: 'math vs code, claims and honesty, compile and style' },
    { title: 'Fix', detail: 'apply confirmed findings, recompile' },
  ],
}

const REPO = '/home/thanhnc/gauss_ws/src/uags'
const FACTS = REPO + '/docs/reports/2026-09-29-paper-revision-1/logs/fact_sheet.md'
const SCR = '/tmp/claude-1005/-home-thanhnc-gauss-ws-src/3b1845f5-8d47-44b8-8bac-e73827ef266b/scratchpad/paper'
const TECTONIC_DIR = '/home/thanhnc/gauss_ws/third_party/tectonic'
const MAIN_MD5 = '689e608cf0020780d3933df3774cf48d'
const KEYS = 'kerbl2023 mildenhall2020nerf radford2021 wilson2025css vegabrown2014 gan2020bayesian kim2025e2bki chopra2026physgs maggio2025bayesian lee2026lamp nagami2025vista mardia2000directional banerjee2005vmf friston2010 parr2022active ye2024gsplat qin2024langsplat zhou2024feature wu2024opengaussian chen2024marepo brachmann2023ace chu2026semantic glocker2013 dai2017scannet wilson2025latentbki joseph2025gradbp zang2026normlift mardia1976bayesian cheng2025occam shi2024legaussians sheppard2026slimvdb jiang2026mif chen2025activesgm lu2025gscd ackermann2025clsplats zeng2025gaussianupdate he2026vgmapping wald2020rio10 wald2019rio straub2019replica zhi2021semanticnerf lan2024clearclip zhou2022maskclip cherti2023reproducible li2022lseg kish1965survey guo2017calibration ilg2018uncertainty kendall2017uncertainties'

const CONTEXT = `
You are helping write the RA-L paper "UAGS" (repo ${REPO}; current draft papers/main.tex; bibliography papers/ref.bib).
READ FIRST: the fact sheet ${FACTS} (method as implemented, honesty and scope rules, style rules, planned experiments) and the
current draft papers/main.tex. The code under usplat/ is the ground truth for every formula.
Allowed \\cite keys (papers/ref.bib): ${KEYS}.
Macros available in the draft: \\todo{...}, \\ph (numeric placeholder), \\R, \\xv, \\zv, \\muv, \\etav, \\Sph. Keep numbers as \\ph unless the fact sheet gives them.
Style: plain simple English, short sentences, no "---" dash chains, no semicolon chains; equations numbered only when referenced.
RULES: never write under docs/reports; PYTHONDONTWRITEBYTECODE=1 for any python; tectonic (LaTeX) is at ${TECTONIC_DIR} (find the binary there).
`

const DRAFTS = [
  { key: 'front', out: 'front.tex', task: `Draft the title (keep "Gaussian Splatting" and the change/uncertainty theme; no "localization"), the abstract (<= 200 words; last sentence \\todo{headline result}), the introduction (<= 0.8 page) and the contribution list (3-4 items, honest novelty per the fact sheet section C: the directional posterior with VIEW-level evidence and a separated epistemic/aleatoric read-out; calibrated change handling (the statistic is a known residual; the contribution is the label-free per-scene calibration, reset and depth-gated re-admission in the Bayesian bookkeeping); the posterior predictive; implementation: one backward pass for all statistics after joseph2025gradbp). Also the keywords (2-5, no localization). Output LaTeX only (no preamble).` },
  { key: 'related', out: 'related.tex', task: `Draft the Related Work section (<= 0.6 page), 3-4 paragraphs: language-embedded 3DGS and training-free lifting (qin2024langsplat, zhou2024feature, wu2024opengaussian, shi2024legaussians, joseph2025gradbp, cheng2025occam, zang2026normlift as concurrent); probabilistic semantic mapping and BKI (vegabrown2014, gan2020bayesian, wilson2025css, wilson2025latentbki, kim2025e2bki, sheppard2026slimvdb, maggio2025bayesian, chopra2026physgs, kendall2017uncertainties); directional statistics (mardia2000directional, mardia1976bayesian, banerjee2005vmf, lee2026lamp); change detection and long-term maps (lu2025gscd, ackermann2025clsplats, zeng2025gaussianupdate, he2026vgmapping, jiang2026mif, chu2026semantic). Each paragraph ends with how we differ, honestly. No localization or active-perception paragraph. Output LaTeX only.` },
  { key: 'method', out: 'method.tex', task: `Draft the Method section (<= 1.7 pages) that matches the fact sheet section A EXACTLY (check every formula against usplat/vmf_field.py, usplat/predictive.py, usplat/vmf_math.py): preliminaries (splat weight T alpha, define T); features and query space (PCA-64, decode-first query); view-level evidence and sufficient statistics; direction posterior and concentration (joint prior MAP, A_d); epistemic and aleatoric read-out (Kish support, unbiased rho_hat; say why not r_bar or a shrunk norm); calibrated change handling (statistic family with beta and k, the chosen member = feature residual stated plainly, history-conditioned null calibration from the mapping pass, reset, depth-gated re-admission, robust set); predictive P0/P1/P2; implementation note on one backward pass for all channels (credit joseph2025gradbp). Include ONE algorithm box (algorithm + algorithmic) for the per-keyframe update. Keep the figure environments for figs/vmf_sphere.pdf and figs/uncertainty_plane.pdf only if they still fit the method (the figures drafter updates them); drop the gradient-trick figure. Output LaTeX only.` },
  { key: 'experiments', out: 'experiments.tex', task: `Draft the Experiments section (setup, metrics, Tab I Replica, Fig views placeholder, Tab II RIO10, Tab III ablations, timing table, qualitative figure placeholder box), a Limitations paragraph and the Conclusion, following fact sheet section E exactly. Every number is \\ph; say which parts are pending with \\todo (extractor name, RIO10 access). Tables in booktabs, footnotesize, arrows for better direction, a "Reimpl." marker on re-implemented baselines, placeholders only. No localization table. Output LaTeX only.` },
  { key: 'figures', out: 'figures_report.md', task: `Update the TikZ figures in papers/figs (you own papers/figs/*.tex and their PDFs; back up each old .tex/.pdf as *.bak before editing): pipeline.tex (Fig. 1) must show the implemented method: keyframe -> dense features + PCA -> one render + one backward (per-view sums R_v, m_v, depth) -> view-level evidence (w_v, u_v) -> posterior (mu*, kappa) with epistemic (effective views) and aleatoric (rho_hat) -> change test calibrated from the mapping pass -> reset / depth-gated re-admission -> predictive render for text queries. Remove EFE and "robust subset for localization". uncertainty_plane.tex: axes = effective views s and agreement rho_hat; region s < tau_min "undefined"; no evidence-decay trajectory (change is a temporal test, not a region). Compile each standalone figure with tectonic and check the PDFs render (pdftoppm -png if available, look at the image). Report what changed.` },
  { key: 'template', out: 'template_report.md', task: `Prepare the RAS conference class for the first RA-L submission: get the official ieeeconf.cls (IEEE/RAS conference template, e.g. from ras.papercept.net template files or the IEEE conference template; verify the source) into papers/ (you may write papers/ieeeconf.cls only). Write ${SCR}/preamble.tex: \\documentclass[letterpaper, 10 pt, conference]{ieeeconf} with \\IEEEoverridecommandlockouts and \\overrideIEEEmargins as the template requires, the packages the draft needs (amsmath, amssymb, graphicx, booktabs, algorithm, algorithmic, xcolor, hyperref, cite; check ieeeconf compatibility, e.g. amsthm clashes), the draft macros, an anonymous byline ("Anonymous Authors" only), and how keywords are written in ieeeconf (no IEEEkeywords environment there). Compile a minimal test document with tectonic and ref.bib to prove it works. Report.` },
]

phase('Draft')
const drafts = await parallel(DRAFTS.map(d => () =>
  agent(`${CONTEXT}\nYour label: draft-${d.key}. Write your output to ${SCR}/${d.out} (create the folder). Do not edit papers/main.tex or papers/ref.bib.\nTASK: ${d.task}\nFinal answer: the path of your output, a summary, and anything you could not verify.`,
    { label: `draft:${d.key}`, phase: 'Draft' }).then(r => ({ key: d.key, out: d.out, report: r }))))
const okD = drafts.filter(Boolean)

phase('Integrate')
const integ = await agent(`${CONTEXT}\nYour label: integrate. Section drafts are in ${SCR}/ (front.tex, related.tex, method.tex, experiments.tex, preamble.tex; figure and template reports). Their reports:\n${okD.map(d => `== ${d.key} ==\n${d.report}`).join('\n')}\n
TASK: assemble the new papers/main.tex.
 1. SAFETY: first check md5sum papers/main.tex == ${MAIN_MD5}. If not, the user edited it: STOP, write your assembled file to ${SCR}/main_new.tex instead and say so.
 2. Copy the current main.tex to ${SCR}/main_before_integration.tex. Move the active-view-selection (EFE) section and the localization table/text of the old draft into papers/parked_sections.tex (with a header comment: parked 2026-09-29, not part of the paper, reasons from the fact sheet).
 3. Write papers/main.tex = preamble + front + related + method + experiments + \\bibliographystyle{IEEEtran}\\bibliography{ref}. Make notation, labels and cross-references consistent across sections; every \\cite key must exist in papers/ref.bib; every \\ref must resolve; keep \\ph/\\todo placeholders.
 4. Compile a COPY (cp -r papers ${SCR}/build) with tectonic; fix errors in papers/main.tex until it compiles with no undefined references or citations. Report page count, overfull boxes, and the PDF path (${SCR}/build/main.pdf). Copy the PDF to ${SCR}/main_v1.pdf.
Do not edit ref.bib (if a key is missing, remove the \\cite and add a \\todo).`, { label: 'integrate', phase: 'Integrate' })

phase('Review')
const FIND = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  where: { type: 'string' }, severity: { type: 'string', enum: ['critical', 'major', 'minor'] }, problem: { type: 'string' },
  evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['where', 'severity', 'problem', 'evidence', 'fix'] } } },
  required: ['findings'] }
const LENSES = [
  { key: 'math', text: 'MATH vs CODE: check every equation, symbol and default in papers/main.tex against usplat/vmf_field.py, predictive.py, vmf_math.py, evaluation.py and the fact sheet. Verify derivations (joint-prior MAP, unbiasedness of rho_hat^2, E[mu] = A_d mu*, the change family and its beta = 0, k = 2 member, the calibration rule). Numerically check any claim you can with small python scripts in your scratch dir.' },
  { key: 'claims', text: 'CLAIMS, HONESTY, SCOPE, ANONYMITY: every sentence that claims novelty or a result must be supported by the fact sheet or marked \\todo/\\ph; check the novelty wording rules (gradient identity credited, NormLift/LatentBKI/CSS/LEGaussians described fairly, the change statistic stated as the known residual), no localization claims, no EFE, no author-identifying text or links, no "first"/"principled"/"without heuristics"/"any rasterizer". Compare the related-work descriptions with docs/reports/2026-09-28-paper-completion-plan/logs/workflow/report_sota.md.' },
  { key: 'form', text: 'FORM: compile papers/ with tectonic (copy to your scratch dir first), report errors/warnings, page count against the 7-page budget, overfull boxes, table column decimals (placeholders fine), figure/table/equation references, undefined \\todo density, readability (plain English, sentence length, no dash chains), and render each page (pdftoppm) and look at the figures for mismatch with the text.' },
]
const reviews = await parallel(LENSES.map(l => () =>
  agent(`${CONTEXT}\nYour label: review-${l.key}. The integrator report:\n${integ}\nREVIEW papers/main.tex (read-only; scratch files only). LENS: ${l.text}\nReport only real problems with evidence.`,
    { label: `review:${l.key}`, phase: 'Review', schema: FIND }).then(r => ({ lens: l.key, ...r }))))
const okRev = reviews.filter(Boolean)

phase('Fix')
const fix = await agent(`${CONTEXT}\nYour label: fix. Apply every critical and major finding (and minor ones when cheap) to papers/main.tex (you may also edit papers/figs/*.tex and papers/parked_sections.tex; not ref.bib). If a finding is wrong, say why. Then compile a copy with tectonic (no errors, no undefined refs/cites), report the final page count, and copy the PDF to ${SCR}/main_v2.pdf.\nFINDINGS:\n${JSON.stringify(okRev, null, 1)}`,
  { label: 'fix', phase: 'Fix' })

return { drafts: okD, integrate: integ, reviews: okRev, fix }
