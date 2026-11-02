import cv2
from pathlib import Path
from modules._common import videolib, imagelib, picturediff as pd

up_path = Path('d:/deepfake/sourcediff/data/cases/20261027T103526264588Z-37d87744/input/WhatsApp_Video_2026-10-27_at_3.33.39_PM.mp4')
src_path = Path('d:/deepfake/sourcediff/library/trusted/video/whatsapp-video-2026-10-27/original.mp4')

timestamps = [float(i) for i in range(10)]
up_frames = list(videolib.iter_frames_at(up_path, timestamps))
src_frames = list(videolib.iter_frames_at(src_path, timestamps))

b_diffs, d_devs, c_diffs = [], [], []
for t, up, src in zip(timestamps, up_frames, src_frames):
    al = pd.align(imagelib.gray(up), imagelib.gray(src))
    if not al['ok']:
        print(f"Align failed at {t}s: {al['reason']}")
        continue
    cmp = pd.compare(up, src, al['H'])
    if not cmp['ok']:
        print(f"Compare failed at {t}s: {cmp['reason']}")
        continue
    b_diffs.append(cmp['block_diff_max'])
    d_devs.append(cmp['detail_dev_max'])
    c_diffs.append(cmp['block_chroma_diff_max'])

print('Max block diff:', max(b_diffs) if b_diffs else None)
print('Max detail dev:', max(d_devs) if d_devs else None)
print('Max chroma diff:', max(c_diffs) if c_diffs else None)
