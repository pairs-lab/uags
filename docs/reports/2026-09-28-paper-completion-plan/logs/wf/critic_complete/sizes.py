# Disk and CPU-time arithmetic for the plan's feature cache (no data read).
GB=1e9
rep_frames = 8*(180+90)          # plan: 8 scenes, 180 map + 90 eval frames
px_rep = 640*480
px_rio = 540*960
print("Replica frames", rep_frames)
print("per-pixel fp16 PCA-64, one Replica frame MB:", px_rep*64*2/1e6)
print("per-pixel fp16 PCA-64, 8 scenes x 180 map frames GB:", 8*180*px_rep*64*2/GB)
print("per-pixel fp16 PCA-64, all 2160 Replica frames GB:", rep_frames*px_rep*64*2/GB)
tok_rep = (480//16)*(640//16)
print("token grid Replica", tok_rep, "tokens; fp16 512-d per frame MB:", tok_rep*512*2/1e6,
      "; all frames GB:", rep_frames*tok_rep*512*2/GB)
# RIO10 update frames: first half of seq02, every 5th (stats: 34415 seq02 frames over 10 scenes)
rio_upd = 34415/2/5
print("RIO10 update frames approx", rio_upd)
print("per-pixel fp16 PCA-64, one RIO10 frame MB:", px_rio*64*2/1e6,
      "; all update frames GB:", rio_upd*px_rio*64*2/GB)
# CPU time: code report 1.81 s/frame at 640x480; scale by pixel count (assumption: linear)
s = 1.81
print("RIO10 update extraction h at 1.81 s/frame:", rio_upd*s/3600)
print("RIO10 update extraction h scaled by pixels:", rio_upd*s*px_rio/px_rep/3600)
