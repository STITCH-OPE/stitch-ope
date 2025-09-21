#!/usr/bin/env bash
# Download/ingest assets into the repo (works with a local zip, Drive file ID/URL, or Drive folder).
# Usage examples (host):
#   sudo make download ZIP_LOCAL=assets.zip
#   sudo make download ZIP_ID=1AbCdEfGhIj...
#   sudo make download ZIP_URL="https://drive.google.com/file/d/<id>/view?usp=sharing"
#   sudo make refresh ZIP_LOCAL=assets.zip
set -euo pipefail

# ---------- Inputs from environment ----------
ZIP_LOCAL="${ZIP_LOCAL:-}"   # path to a local .zip inside /workspace (preferred & private)
ZIP_ID="${ZIP_ID:-}"         # Google Drive file ID (public if share is "Anyone with link")
ZIP_URL="${ZIP_URL:-}"       # Google Drive file URL (public)
GDRIVE="${GDRIVE:-}"         # Google Drive folder URL (fallback)
STAMP="${STAMP:-.cache/download.stamp}"
REFRESH="${REFRESH:-}"

# ---------- Idempotency ----------
mkdir -p .cache
if [[ -f "$STAMP" && -z "$REFRESH" ]]; then
  echo "[skip] assets already downloaded (rm $STAMP or run REFRESH=1 to force)"
  exit 0
fi

# ---------- Tools ----------
python3 -m pip install --no-cache-dir gdown >/dev/null

# ---------- Fetch / Unpack ----------
TMP="$(mktemp -d)"
SRC=""

if [[ -n "$ZIP_LOCAL" ]]; then
  echo "[ingest] local zip: $ZIP_LOCAL"
  mkdir -p "$TMP/unzipped"
  unzip -q "$ZIP_LOCAL" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -mindepth 1 -maxdepth 1 -type d -print -quit || true)"
  [[ -n "$SRC" ]] || SRC="$TMP/unzipped"
elif [[ -n "$ZIP_ID" ]]; then
  echo "[download] file id: $ZIP_ID"
  gdown --id "$ZIP_ID" -O "$TMP/assets.zip"
  mkdir -p "$TMP/unzipped"
  unzip -q "$TMP/assets.zip" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -mindepth 1 -maxdepth 1 -type d -print -quit || true)"
  [[ -n "$SRC" ]] || SRC="$TMP/unzipped"
elif [[ -n "$ZIP_URL" ]]; then
  echo "[download] file url: $ZIP_URL"
  gdown --fuzzy "$ZIP_URL" -O "$TMP/assets.zip"
  mkdir -p "$TMP/unzipped"
  unzip -q "$TMP/assets.zip" -d "$TMP/unzipped"
  SRC="$(find "$TMP/unzipped" -mindepth 1 -maxdepth 1 -type d -print -quit || true)"
  [[ -n "$SRC" ]] || SRC="$TMP/unzipped"
elif [[ -n "$GDRIVE" ]]; then
  echo "[download] folder: $GDRIVE"
  gdown --fuzzy --folder "$GDRIVE" -O "$TMP"
  SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d -print -quit)"
else
  echo "[error] No ZIP_LOCAL / ZIP_ID / ZIP_URL / GDRIVE provided."
  exit 2
fi

echo "[download] source: $SRC"

# ---------- Destinations ----------
mkdir -p \
  opelab/examples/d4rl/models opelab/examples/d4rl/policy \
  opelab/examples/gym/models  opelab/examples/gym/policy  \
  opelab/examples/diffusion_policy/policy \
  dataset

# ---------- Routing rules ----------
shopt -s nullglob

# 1) Model checkpoints (.pth): route by name AND also copy into diffusion_policy/policy
for f in "$SRC"/*.pth; do
  base="$(basename "$f")"
  case "$base" in
    hopper*.pth|walker*.pth|cheetah*.pth)
      dest="opelab/examples/d4rl/models"
      ;;
    pendulum*.pth|acrobat*.pth|acrobot*.pth)
      dest="opelab/examples/gym/models"
      ;;
    *diffusion*.pth)
      dest="opelab/examples/diffusion_policy/policy"
      ;;
    *)
      dest="opelab/examples/d4rl/models"
      ;;
  esac

  # primary copy
  cp -n "$f" "$dest"/

  # extra copy for diffusion workflows (requested)
  mkdir -p opelab/examples/diffusion_policy/policy
  cp -n "$f" opelab/examples/diffusion_policy/policy/
done

# 2) Policy files at top level (e.g., .pkl)
for f in "$SRC"/*.pkl; do
  base="$(basename "$f")"
  case "$base" in
    *diffusion*.pkl) dest="opelab/examples/diffusion_policy/policy" ;;
    pendulum*.pkl|acrobat*.pkl|acrobot*.pkl) dest="opelab/examples/gym/policy" ;;
    *) dest="opelab/examples/d4rl/policy" ;;
  esac
  mkdir -p "$dest"
  cp -n "$f" "$dest"/
done

# 3) Policy subfolders: SRC/policy/<name>/*
if [[ -d "$SRC/policy" ]]; then
  for d in "$SRC/policy"/*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    case "$name" in
      *diffusion* ) dest="opelab/examples/diffusion_policy/policy" ;;
      pendulum|acrobat|acrobot) dest="opelab/examples/gym/policy" ;;
      *) dest="opelab/examples/d4rl/policy" ;;
    esac
    mkdir -p "$dest/$name"
    cp -rn "$d/." "$dest/$name/"
  done
fi

# 4) Directly packaged diffusion tree: SRC/diffusion_policy/policy/*
if [[ -d "$SRC/diffusion_policy/policy" ]]; then
  for d in "$SRC/diffusion_policy/policy"/*; do
    [[ -d "$d" ]] || continue
    name="$(basename "$d")"
    mkdir -p "opelab/examples/diffusion_policy/policy/$name"
    cp -rn "$d/." "opelab/examples/diffusion_policy/policy/$name/"
  done
fi

# 5) Datasets
if [[ -d "$SRC/dataset" ]]; then
  cp -rn "$SRC/dataset/." opelab/examples/gym/dataset/
fi

# ---------- Summary ----------
touch "$STAMP"
echo "[done] d4rl models:";      ls -1 opelab/examples/d4rl/models || true
echo "[done] gym  models:";      ls -1 opelab/examples/gym/models  || true
echo "[done] d4rl policy:";      ls -1 opelab/examples/d4rl/policy || true
echo "[done] gym  policy:";      ls -1 opelab/examples/gym/policy  || true
echo "[done] diffusion policy:"; ls -1 opelab/examples/diffusion_policy/policy || true
