# Disk budget and 3DGS/field sizes for the revised UAGS plan. [A] = assumption.
GB=1e9
zip_replica=6762780327            # WF/judge/dbx_head.txt
ext_factor=1.3                    # [A] extracted/zip, not checked
tok_rep=40*30*512*2               # token grid fp16, 640x480 / 16 px
tok_rio=34*60*512*2               # 540x960 portrait -> ceil(33.75)=34 x 60 tokens [A: model keeps native grid]
frames_rep=2160
upd=34415/2/5                     # RIO10 seq02 first half, stride 5 (rio10_stats_summary.txt: 34,415 frames)
evl=34415/2/10                    # seq02 second half, stride 10 (semantic block)
gs_bytes=59*4                     # 3DGS params per Gaussian with SH deg 3
cap=1_000_000                     # [A] Gaussian cap per map
field=cap*(64+1)*4                # one channel, d=64
print(f"Replica zip {zip_replica/GB:.2f} GB; extracted [A x{ext_factor}] {zip_replica*ext_factor/GB:.2f} GB; unzip peak {zip_replica*(1+ext_factor)/GB:.2f} GB")
print(f"Replica token cache {frames_rep*tok_rep/GB:.2f} GB ({tok_rep/1e6:.2f} MB/frame)")
print(f"RIO10 token per frame {tok_rio/1e6:.2f} MB; update {upd:.0f} + eval {evl:.0f} frames = {(upd+evl)*tok_rio/GB:.2f} GB for all 10 scenes; per scene {(upd+evl)/10*tok_rio/GB:.2f} GB (+ seq01 frames, not counted)")
print(f"3DGS map at cap {cap}: {cap*gs_bytes/GB:.3f} GB; 18 maps {18*cap*gs_bytes/GB:.2f} GB")
print(f"field one channel d=64 at cap: {field/GB:.2f} GB; raw+centred+2 EWMA = 4 channels {4*field/GB:.2f} GB (GPU persistent)")
print(f"MASt3R depth png 540x960x2B uncompressed {540*960*2/1e6:.2f} MB/frame; update frames all scenes {upd*540*960*2/GB:.2f} GB")
