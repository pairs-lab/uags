# Attempt 01: 2D dense-CLIP capacity gate on Replica (plan K8)

## Idea
Before any mapping run, check that training-free dense CLIP separates Replica classes: validation
scenes office_0 and room_2, 20 frames each, classify every labelled pixel among the classes present;
pass bar (plan, an assumption) >= 15 mIoU on both scenes. Variants: per-pixel, FastSAM mask pooling,
per-pixel through PCA-64 with the decode-first query; templates.

## Result
Run 1 ([../logs/check_clip2d_run1_brokenextractor.log](../logs/check_clip2d_run1_brokenextractor.log)): per-pixel mIoU 0.4 / 0.6, pixel accuracy
1.0 / 4.5% -- below chance (1/24 classes = 4%). Diagnosis: open_clip 3.3's transformer is
batch-first; usplat/clip_dense.py permuted to (L, N, D), so every token was its own batch item and
no attention across tokens happened (checked: `transformer.batch_first == True`). Fixed; last-block
variants on 8 frames each ([../logs/maskclip_variants_fixed.json](../logs/maskclip_variants_fixed.json)):
| last block | office_0 mIoU / acc | room_2 mIoU / acc |
|---|---|---|
| residual + FFN | 9.8 / 27.9 | 18.8 / 59.8 |
| value only (MaskCLIP) | 11.1 / 39.0 | 21.9 / 64.3 |
| q-q attention (ClearCLIP) | 11.2 / 47.1 | 21.6 / 69.9 |
Run 2, ClearCLIP default, 20 frames ([../logs/check_clip2d_run2.log](../logs/check_clip2d_run2.log)): best per-pixel
13.4 / 51.5 (office_0), 22.6 / 73.0 (room_2) with "a photo of a {} in a room."; PCA-64 +
decode-first 13.1 / 22.2 (no loss); FastSAM mask pooling 11.3 / 17.3 (worse); 5-template
ensemble 12.5 / 21.4. Run 3, openai weights ([../logs/check_clip2d_run3_openai.log](../logs/check_clip2d_run3_openai.log)):
best worst-scene 13.4 (ensemble), room_2 19.0 -- no gain.

## Critic
Bad for the gate: worst-scene 13.4 < 15 with both pre-registered fallbacks (mask pooling, openai)
exhausted; the bar is NOT moved after the fact. Good for the pipeline: a silent extractor bug that
invalidated every earlier real-CLIP number was found and fixed; a real-feature regression test
(tests/test_clip_real.py, 69.1% pixel accuracy) now guards it. Frozen on validation: laion2b,
ClearCLIP last block, template "a photo of a {} in a room." (configs/vocab_replica.json).
