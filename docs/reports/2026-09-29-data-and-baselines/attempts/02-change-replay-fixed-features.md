# Attempt 02: the 7-Scenes change replay re-run with correct CLIP features

## Idea
Repeat the method run's G1 replay (simulated swaps, chess = tuning, office = unseen) now that the
extractor works; check whether yesterday's design choices (centred channel, beta = 0.9) still hold.

## Result
Chess ([../logs/g1_chess_fixed.json](../logs/g1_chess_fixed.json)), change on the last half of each point's views:
old 1 - r_bar 0.812, feature residual (2 views) 0.811, raw Kish u_ale 0.813; our change score
c_max 0.781-0.793 (raw) and 0.781-0.787 (centred); centred u_ale 0.735; normalised c 0.656.
Office once, frozen raw beta = 0.5 ([../logs/g1_office_fixed.json](../logs/g1_office_fixed.json)): u_ale 0.892, 1 - r_bar
0.890, residual 0.872, ours c_max 0.857.
Long-history regime (change on the last 30% / 20% of views; [../logs/](../logs/) g1_*_fixed_frac*.json):
at 20% dispersion drops (chess 0.742, office 0.833) while the temporal scores hold (ours 0.789 /
0.835, residual 0.803 / 0.861).

## Critic
Bad for the method as designed: with correct features the centred channel HURTS (reverses the
broken-feature finding) and the two-bucket score does not beat the simple residual. Dispersion is
best when half of the views changed; temporal tests are best when the change is a small part of
the history (the realistic map-then-update regime). Centring is now OFF by default.
