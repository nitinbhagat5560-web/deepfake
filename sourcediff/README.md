# SourceDiff: Image and Video comparison (Path 1 rehearsal)

**Two doors: Image and Video.** SourceDiff looks for a trusted original and compares the upload with it. Standalone
audio files are rejected; an audio track inside a video may be analyzed locally for transcript and lip-sync evidence.
This is a rehearsal build, not a general deepfake detector.

**Current safety status**
- Exact byte-for-byte video copies can be identified by SHA-256.
- Non-exact video matching is currently disabled because its Gate 1 calibration report is not present.
- Video picture, transcript, and sync ranges are uncalibrated. The engine fails closed instead of treating them as agreement.
- Image matching currently uses the tiny 32x32 grayscale stand-in. Its rehearsal report uses sample images, not a representative real-media test set.
- There is **no Path B** (detectors) and **no Gate 2** (known fakes). A file with no trusted original gets "Can't Verify".
- This project copy includes trusted library entries and their metadata. FAISS indexes and local model files are derived/local data, not required source files.

---

## 1. Run it (10 minutes)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows:  .venv\Scripts\activate
pip install -r requirements.txt        # needs Python 3.10+
                                       # requirements-tested.txt = the exact versions it was tested with
# Optional SSCD matching: install PyTorch using pytorch.org's selector, then:
python -m pip install -r requirements-sscd.txt
python tools/download_sscd.py           # downloads the official TorchScript checkpoint locally

python -m pytest                       # Use "python -m pytest", not plain "pytest"
python tools/rehearsal.py              # runs 8 demo files end to end, lists every file a run writes. Ends with: ALL OK
python -m uvicorn sourcediff.api:app --reload      # then open http://127.0.0.1:8000
```

The first transcript comparison may download the default `small` faster-whisper model. Set
`SOURCEDIFF_WHISPER_MODEL` to a local faster-whisper model directory to run without a model download. Test and rehearsal
commands do not need to invoke transcription. Rebuild derived indexes with `python tools/library_cli.py build-index`
after adding media or changing an embedder. If library/demo media are absent, `python tools/seed_demo.py` creates the
rehearsal image set. After `rehearsal.py`, inspect `data_rehearsal/cases/<case id>/`.

## 2. What happens to an upload (Path 1)

```
upload -> INGEST -> GATE 1 -> ALIGN -> IMAGE/VIDEO CHANNELS -> VERDICT -> CARD + HISTORY
```

| Step | File | What it does | Image | Video |
|---|---|---|---|---|
| Ingest | `sourcediff/ingest.py` | SHA-256, media door, duration, audio/face flags | image | video; standalone audio rejected |
| Gate 1 | `modules/g1_visual_match` | retrieve a trusted source and estimate upload coverage | active image embedder | exact SHA-256 active; non-exact candidate matching disabled pending calibration |
| Align | `modules/aligner` | fit image registration or video timeline; report jumps with upload times | image alignment | timeline fit and cut/jump detection |
| Picture | `modules/diff_picture` | compare image regions or aligned video frames | image metrics | sampled once per second; each evidence row carries upload timestamps |
| Transcript | `modules/diff_transcript` | not applicable | — | local faster-whisper words/timestamps, true word error rate, and waveform correlation |
| Sync | `modules/diff_sync` | not applicable | — | per-second paired mouth measurements and audio-visual correlation; skipped if it cannot measure them |
| Verdict / Card | `sourcediff/verdict.py`, `sourcediff/report.py` | re-judge metrics from config and explain outcomes | Authentic only under Path 1 rules | exact copies may be Authentic; non-exact outcomes are Edited, Matched — no edit found, or Can't Verify (incomplete) |

**SHA-256 vs FAISS.** SHA-256 proves exact byte identity. FAISS searches visual descriptors to retrieve a possible source;
it never receives the file hash. The current image embedder is `tiny32_gray_v1`; SSCD is available as an optional local
backend but is not enabled by default.

**Non-exact video outcomes.** A jump or any measured channel outside its calibrated range gives **Edited**, with the
channel, metric, value, upload-time interval, and available comparison images. If every applicable channel is measured
and inside its calibrated range, the result is **Matched — no edit found**; this wording is deliberately not "Authentic".
If a check or any required portion is missing, uncalibrated, or cannot be measured, the result is **Can't Verify
(incomplete)** with the missing check and reason. In this checkout, non-exact video Gate 1 and the video diff ranges are
uncalibrated, so the video path remains fail-closed until matching reports are produced.

### The Golden Rule
An exact, byte-identical trusted copy may be called **Authentic**. A non-exact image can be called Authentic only when
Gate 1 covers the upload and all applicable calibrated channels agree. A non-exact video is never labeled Authentic by
this build: it receives one of the three outcomes above. Missing, skipped, errored, or uncalibrated evidence never counts
as agreement. The engine re-judges measurements against `config/genuine_ranges.json`; it does not trust a module's verdict.
No-match cards say: *"No original found. This does not mean the media is genuine."*

`tests/test_golden_rule_fuzz.py` throws 500 random cases at the engine and compares it with a second, separately
written version of the rule. I also broke the engine on purpose three ways (an unchecked channel counted as agreeing,
partial coverage counted as full, a module's opinion trusted) and the tests failed each time, so they do catch it.

## 3. Where things are

```
config/            registry.json (modules, owners)   verdict.json (rules)   genuine_ranges.json   gates.json
sourcediff/        contract.py (the shared format: FREEZE IT)   module_api.py   runner.py   verdict.py
                   ingest.py   report.py   history.py   audit.py   api.py   ui/index.html   paths.py
modules/<name>/    run.py   one folder per model; modules never call each other
modules/_common/   imagelib.py (image embedder + FAISS)   videolib.py (sampled-video index)   picturediff.py (align + compare)
library/           trusted/{image,video}/<id>/{original.ext, meta.json}   index/ (derived: rebuild it)
data/              cases/<case id>/ (every upload + every file its run wrote)   history.db   (made at first use)
                   history.db also stores submissions, duplicate-result cache, and a verified SQLite registry for disk-backed library media
tools/             rehearsal.py  seed_demo.py  calibrate_image.py  library_cli.py  run_case.py  _variants.py
tests/             one file per concern; hand-written evidence, no models needed
demo/media/        the 8 demo uploads        reports/calibration_image.json   the measured numbers
```

Set `SOURCEDIFF_DATA`, `SOURCEDIFF_LIBRARY`, `SOURCEDIFF_CONFIG` to point somewhere else (a second disk, a shared drive).

### MySQL metadata database

SourceDiff can replace the SQLite history database while keeping uploaded media and generated artifacts on disk.
Set these environment variables before starting FastAPI:

```powershell
$env:SOURCEDIFF_DB_BACKEND="mysql"
$env:SOURCEDIFF_MYSQL_HOST="127.0.0.1"
$env:SOURCEDIFF_MYSQL_PORT="3306"
$env:SOURCEDIFF_MYSQL_DATABASE="sourcediff"
$env:SOURCEDIFF_MYSQL_USER="sourcediff_app"
$env:SOURCEDIFF_MYSQL_PASSWORD="<your-secret-password>"
```

The migration script creates the MySQL tables and copies users, submission history, duplicate cache entries,
and library registry metadata from the existing SQLite file:

```powershell
python tools/migrate_sqlite_to_mysql.py --sqlite data/history.db
```

It deliberately stops if a target MySQL table already has data. Media files, case directories, and library files are not
duplicated into MySQL; their SHA-256 hashes, paths, cards, and metadata are stored in the database instead.

**I want to... open this**

| I want to | Open |
|---|---|
| understand one verdict | `data*/cases/<id>/card.md`, then `verdict.json` (every number and its genuine range) |
| change a rule or a label | `sourcediff/verdict.py`, `config/verdict.json`; then `python -m pytest` |
| see how a module is written | `modules/diff_picture/run.py` (image and video) |
| add originals to the library | section 6 |
| change what counts as "genuine" for a number | `config/genuine_ranges.json` (measure it, do not guess: section 7) |
| change the page | `sourcediff/ui/index.html` (one file, no build step) |
| change what is saved per upload | `sourcediff/history.py` |

## 4. Work plan for two people, 10 days

Write your names in `config/registry.json` -> `"owner"`. One person per module folder means no merge fights.

- **Person A, "find the original":** Gate 1 for video, the aligner, the library, Gate 1 calibration.
- **Person B, "compare with it":** ingest, the picture-diff channel, the page and history, channel calibration.

Both: Day 1 together, Day 6 together (calibration), Day 8 together (dry run).

| Day | Person A | Person B | Done when |
|---|---|---|---|
| 1 | Run everything. Read `contract.py`, `module_api.py`, `runner.py`, `verdict.py`, then `g1_visual_match/run.py`. Collect trusted video originals (public domain or official, 30 to 120 s; URL and licence in `meta.json`). | Same reading, then `diff_picture/run.py` and `picturediff.py`. Make test material: re-uploads (WhatsApp, YouTube, re-encode) and consented edited copies of your own clips. | Both can explain a verdict from the files in `data_rehearsal/` |
| 2 | **Gate 1 video:** calibrate the sampled-frame FAISS stand-in against genuine copies and hard negatives. | Calibrate video picture comparison on genuine re-encodes and edited clips. | A 30 s cut of a library video matches with coverage 1.0; an unrelated video does not match |
| 3 | Build hard-negative and holdout test sets for video matching. | Measure false matches and missed matches; do not enable a threshold unless results are separable. | Video thresholds are measured, not guessed |
| 4 | **Aligner for video:** video line fit and jump detection are implemented; test with cut clips. | Test picture comparison on videos with and without audio tracks; the audio track must not affect results. | A clip with a cut-out middle is reported "Edited / spliced" |
| 5 | Calibrate the Gate 1 threshold on real clips (copy `tools/calibrate_image.py`). Test partial cuts. | Validate that videos with an audio stream and silent videos follow the same visual path. | The pipeline judges video frames only |
| 6 | Together: measure `genuine_ranges.json` for every channel on real re-uploads; set `calibrated: true` and `n`. Run `python -m pytest`. | | No `calibrated: false` left for a channel you demo |
| 7 | Long videos: run the pipeline as a background job (`api.py` runs it inside the request today). | Segments/timeline on the card; test-set script that prints N and the 3/N bound. | A 2-minute video does not time out the page |
| 8 | Together: full dry run of your demo script with a stopwatch. Fix what breaks. | | You can run the demo twice in a row |
| 9-10 | Freeze. Write down what you will rebuild at the hackathon. Rest. | | |

If time is short, cut in this order (a cut only ever moves results toward "Can't Verify", never toward "Authentic"):
Keep: Gate 1 video, aligner, picture comparison.

## 5. How to build a module

1. Open its `run.py`. The docstring says what to compute, with which tool, and when it is done.
2. Read `modules/diff_picture/run.py` (real) and `sourcediff/module_api.py`. A module is one function
   `run(ctx) -> ModuleResult` plus `run_module(NAME, run)`. It reads `ctx.input_path` and earlier results and writes
   evidence rows: a number, its unit, timestamps, plain-language limits. **It never decides a verdict.**
3. Run it by hand: `python tools/run_case.py FILE`, or `python modules/<name>/run.py data/cases/<case id>`, then open `results/<name>.json`.
4. Test the engine's reaction before the model exists: write `<case>/scenario/<name>.json`, for example
   `{"evidence": [{"metric": "logmel_distance", "value": 0.9}]}`, and re-run (`tests/test_runner.py` shows how).
   Once a case has a `scenario/` folder, **every** real module steps aside for that case, so a dummy file never reaches a real model.
5. Measure its genuine range (section 7), put it in `config/genuine_ranges.json` with `calibrated: true` and `n`,
   set `"implemented": true` and the door in `"implemented_doors"` in `config/registry.json`, run `python -m pytest`.

Until a channel's range is calibrated, the engine will **not judge it** (`allow_uncalibrated: false` in `config/verdict.json`),
so that channel counts as "did not run". While you are still developing a channel you may set it to `true` to see
your numbers judged; the card then carries a loud warning. Never leave it `true` for a demo.
Video picture metrics use separate `video_*` range keys. The image ranges do not validate video: collect genuine video
re-encodes and edited clips, measure those metrics and the video Gate threshold, then update their calibration metadata.

The runner starts each module as its own process. A crash becomes `status: "error"`, which the engine treats as "did not run".
A module that needs an old torch (SyncNet) can have its own virtualenv: set `"python"` in the registry.

## 6. Adding library media

```bash
python tools/library_cli.py add-folder ~/Downloads/originals --publisher "..." --license "..."   # bulk
python tools/library_cli.py add-trusted clip.mp4 --id clip-01 --title "..." --url "..." --license "..."
python tools/library_cli.py build-index      # ALWAYS after adding. The index is derived: never share or merge it
python tools/library_cli.py list
```

The subfolder (`image` / `video`) is chosen from the file extension. Standalone audio files are unsupported. Videos with
audio tracks are stored unchanged; transcript and lip-sync analysis may use that audio locally. `build-index` indexes images and samples trusted
videos at 1 frame/s. With the default tiny video embedder, each indexed frame also includes overlapping spatial crop
views so cropped uploads can retrieve the same source. Rebuild the video index after updating to include these views.
Gate 1 uses the configured per-door embedder. The video threshold is deliberately marked uncalibrated:
non-exact candidate matches do not proceed to alignment until genuine video re-encodes and hard negatives have been
measured and the threshold marked calibrated.
To share the library between you two, sync `library/trusted/`
(Google Drive for desktop, or Git LFS) and each run `build-index`. Keep `data/history.db` out of any synced folder.
Only use media you have the right to use; keep the URL and licence in `meta.json`.

### Optional SSCD matching

SSCD runs locally; it does not call a paid inference API. Install PyTorch using
[the official installation selector](https://pytorch.org/get-started/locally/), install `requirements-sscd.txt`,
then run `python tools/download_sscd.py` to store the official TorchScript checkpoint under `library/models/`.
Set `SOURCEDIFF_SSCD_MODEL` to use a checkpoint at another path.

To try it for images, change `config/gates.json` -> `image.embedder` to `sscd_disc_mixup`, then run:

```bash
python tools/library_cli.py build-index
python tools/calibrate_image.py
python tools/calibrate_image.py --write
```

Review the calibration output before `--write`: it updates the image matching threshold and image genuine ranges.
Until calibrated for SSCD, matching is fail-closed and the old thumbnail threshold is not reused. Rebuild indexes
after changing embedders. SSCD also works for sampled video frames by changing `video.embedder`, but video matching
remains fail-closed until genuine video re-encodes and hard negatives are measured and a threshold calibrated
specifically for SSCD. Image calibration does not calibrate video. Exact byte-identical video verification is unchanged.

Scaling later: media files go to object storage (S3, MinIO), the records and history to PostgreSQL, fingerprints to
FAISS or pgvector. That is a plan, not something built here.

## 7. Calibration: how a number becomes evidence

A model output means nothing until you know what it looks like on **genuine** media. For each channel: re-upload
your genuine clips the way real users do, measure the number, set the genuine range from that, then test on edited
copies. `tools/calibrate_image.py` is the worked example (about 30 seconds): it measures on some pictures and **tests on
held-out ones**. Always report **N**. If 0 of N edits slipped through, the true miss rate is still only bounded below about
**3/N** at 95% confidence. Say "0 of N", say N, say the bound. Never "100%".

### What the image door measured (`reports/calibration_image.json`; run `python tools/calibrate_image.py` to reproduce)
Setup: 15 scikit-image sample pictures; genuine = shrink (0.5 to 1.0) x JPEG (30 to 90); edits = pasted patches, regional
blur, inpainting, copy-move, text, iso-luminant recolour (Lab a*/b* shift), each then re-uploaded; 10 pictures set the ranges, 5 held out.
Grey-only compare missed brightness-matched recolours (e.g. hair dye); `block_chroma_diff_max` closes that hole.

| Measurement | Result |
|---|---|
| Gate 1: genuine re-uploads vs other pictures | lowest genuine similarity 0.996 (n=160), highest stranger 0.536 (n=63), so threshold 0.766 sits in the gap |
| Picture ranges (largest genuine value seen, no slack) | `block_diff_max` up to 2.63 grey levels, `detail_dev_max` up to 2.94, `block_chroma_diff_max` up to 2.81 Lab a*b*, from 160 genuine re-uploads |
| Held-out genuine re-uploads called "edited" | **0 of 77** (23 more could not be aligned: low-texture pictures, answered Can't Verify) |
| Held-out edits that would be called **Authentic** | **8 of 150** (5.3%) |
| Held-out edits caught by the picture check | 109 of 150; 33 more stopped earlier as Can't Verify (match lost, or no alignment) |
| By type, edits passed as Authentic | blur 60 px 2/10, blur 100 px 2/10, blur 150 px 1/10, inpaint 60 px 1/10, copy-move 60 px 1/10, text 1/10; **all patches, all recolours, inpaint 100 and copy-move 100: 0 passed** |

**What this means.** The picture check is good at pasted, written and recoloured content and weak at small blurred or removed regions, which
change little in smooth areas. So **do not claim "0 false-Authentic" for the picture check**. Say: "on a held-out set of 150 edits,
8 passed; the misses are small blur/inpaint edits in smooth areas. Iso-luminant recolour (hair-dye style) is caught by chroma."

**Honest caveats on these numbers**
- Genuine means *shrink + JPEG only*. Real platforms also sharpen, change colour, rotate by EXIF. Add those to your genuine set before trusting a range.
- Slack: the upper range edge is the largest genuine value seen x 1.0. Adding 25% slack let 9 of 150 edits pass instead of 8
  (the tool prints the full table). A genuine copy called "edited" is the safe error; an edit called "Authentic" is the dangerous one,
  so the default has no slack. That choice was made after seeing the held-out table, so those numbers are no longer a clean test: for a headline claim, use new pictures.
- Sample photos from one package, one image size (about 800 px). Redo everything on real media.
- Why the picture check is built the way it is (`modules/_common/picturediff.py`): comparing pixels after the first
  alignment flagged half-size genuine re-uploads as edited, because the alignment was off by a pixel or two. Refining the alignment
  (ECC, sub-pixel), shrinking the *original* to the upload's size and blurring both a little took the largest genuine value from about 35 down to 2.6 grey levels. A metric that cannot separate genuine from edited must never be given a range and shipped.

## 8. Known limits (say them out loud before anyone else does)

- **No login.** `user_id` is whatever the page sends, so history is per name, not per person.
- The pipeline runs **inside the web request**: fine for images, not for long video.
- Uploads stay in `data/cases/` until you delete them. A real product needs a consent line and a retention rule (check India's DPDP Act applies to you).
- The default 32x32 thumbnail matcher does **not** survive crops or rotations: `demo/media/known_limit_crop_astronaut.jpg` shows it (Can't Verify). SSCD is optional and not active by default; it needs PyTorch, a downloaded checkpoint, a rebuilt index, and a fresh calibration.
- Low-texture pictures (clock, cell micrographs) cannot be aligned; they get Can't Verify. That is safe, and it is a coverage limit.
- Only image/video are supported doors. Standalone audio uploads are rejected; local transcript and sync modules operate only on a video's soundtrack.
- Transcript and sync ranges have no measurement reports and are marked uncalibrated. They cannot agree until calibrated.
- Video Gate 1 and video picture ranges have no matching calibration report and are disabled/uncalibrated. Non-exact video matching cannot currently run in the default configuration; exact video identity remains available.
- Local faster-whisper compares recognized words, not meaning. A negation or other semantic flip may be missed; there is no NLI/meaning channel.
- Video picture/sync checks sample approximately once per second; they are not a frame-by-frame guarantee.
- Hindi strings on the card are a draft for a native speaker to review.
- The image calibration report is based on scikit-image sample photos and must not be presented as real-media performance. Its report records 8 of 150 held-out edits passing as Authentic; do not claim zero misses.
- The history chain is tamper-**evident**, not tamper-proof: someone who rewrites the whole chain can still do it. Anchor the newest hash somewhere you do not control.

## 9. API (for the page or a teammate's script)

`POST /api/analyze` (file + `user_id`) returns the card and the history row it created. `GET /api/history?user_id=` newest first
(no user = everyone). `GET /api/history/verify` re-checks the SHA-256 chain. `POST /api/forget` blanks a user's name and
file names. `GET /api/case/{id}/card`, `GET /api/case/{id}/file/{card.md|card.json|verdict.json|diff_picture_map.png|...}`.
`GET /api/health`. Interactive docs: http://127.0.0.1:8000/docs

## 10. What to carry to the hackathon (you rebuild from scratch there)

Reuse the **ideas and the order**, not the demo code:
1. `contract.py`: one evidence format; modules never call each other; the runner treats a failure as "did not run".
2. The verdict rules, the Golden Rule, and the random test that proves it (with the oracle written separately).
3. The calibration method: measure on genuine re-uploads, **test on held-out data**, report N and the 3/N bound.
4. The library layout (`trusted/{image,video}`), the derived index, and "coverage is measured on the upload".
5. The lessons: pixel comparison fails on genuine re-uploads until registration is refined; "0 of N" is not "never wrong"; a broken probe or an unbuilt channel must lower the verdict, never raise it.

## Troubleshooting

- `ModuleNotFoundError: sourcediff` -> run from the project folder with `python -m pytest` / `python tools/...`.
- Everything says "Can't Verify" -> the library index is missing: `python tools/library_cli.py build-index`.
- `faiss` will not install -> `pip install faiss-cpu` (Python 3.10 or newer).
- Port busy -> `python -m uvicorn sourcediff.api:app --port 8001`.
- `ffprobe` not found -> install ffmpeg (needed for video files, not for images).
