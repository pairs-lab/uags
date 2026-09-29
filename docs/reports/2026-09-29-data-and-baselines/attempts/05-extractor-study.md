# Attempt 05: trained / stronger dense extractors (user asked for better features)

## Idea
Replace training-free ClearCLIP (worst-scene 13.4 mIoU, capacity gate failed) with a trained or stronger
dense vision-language extractor. Four candidates integrated behind usplat/extractors (one module each),
scored with the unchanged frozen protocol (scripts/check_clip2d.py: validation scenes office_0 + room_2,
20 frames each, classes present, three prompt templates, PCA-64 decode-first variant). Rule fixed before:
best worst-scene mIoU; tie-break mean mIoU, then speed; the PCA-64 path matters because the map stores it.

## Result
| extractor | worst scene (per pixel) | worst scene (PCA-64) | office_0 / room_2 (best template) | log |
|---|---|---|---|---|
| LSeg (ICLR 2022; ViT-L/16 + CLIP ViT-B/32 text; vendored, 99.97% argmax agreement with the official code) | 23.2 | 23.2 | 23.2 / 33.5 | [lseg_check.json](../logs/extractors/lseg_check.json), [vs official](../logs/extractors/lseg_vs_official.log) |
| dino.txt (CVPR 2025) | 23.7 | 22.7 | 24.1 / 23.7 | [dinotxt_check.json](../logs/extractors/dinotxt_check.json) |
| Talk2DINO (ICCV 2025) | 21.7 | -- | 22.7 / 21.7 | [talk2dino_check.json](../logs/extractors/talk2dino_check.json) |
| ProxyCLIP (ECCV 2024, laion2b + DINOv2-reg, 448) | 20.3 | -- | 20.3 / 27.0 | [proxyclip_check_s448.json](../logs/extractors/proxyclip_check_s448.json) |
| ClearCLIP (current) | 13.4 | 13.1 | 13.4 / 22.6 | [../logs/check_clip2d_run2.log](../logs/check_clip2d_run2.log) |
LSeg pixel accuracy 78.0-85.2%; the check ran in 37 s on the GPU for 80 frame extractions.

## Critic
Good: +9.8 points worst-scene over ClearCLIP; the plan's capacity gate (>= 15) now passes. Chosen: LSeg with
the 5-template ensemble. dino.txt is 0.5 higher on per-pixel worst scene (within noise) but lower on the
PCA-64 path the map uses (22.7 vs 23.2) and on the mean (23.9 vs 28.4); LSeg is also the extractor of CSS
and Feature 3DGS, so baseline comparisons stay like-for-like. Caveats: LSeg is trained on labelled
segmentation data (a closed-set supervision signal; open-vocabulary generalisation beyond those concepts is
weaker than CLIP's); the four agents stopped at a session limit before their judge ran, so the choice and
the protocol check were done by the lead (all JSONs: 20 frames per scene, both validation scenes, the
unchanged script). Only validation scenes were used.
