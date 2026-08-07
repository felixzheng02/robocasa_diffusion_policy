#!/usr/bin/env bash
# Fetch graspnet-baseline's pretrained weights.
#
# Upstream hosts these only on Google Drive and Baidu Pan. The Drive copies are heavily
# downloaded and Drive periodically refuses programmatic access to them entirely
# ("Too many users have viewed or downloaded this file recently"), which no amount of
# retrying or cookie juggling fixes -- it is a per-file quota on Google's side.
#
# So this script tries, and tells you exactly what to do by hand if it cannot. It is not a
# fallback path worth automating further: a browser download takes ten seconds.
#
# Usage:  ./fetch_checkpoint.sh [rs|kn]

set -euo pipefail

WHICH="${1:-rs}"
DEST_DIR="${CKPT_DIR:-/home/felix/Desktop/robocasa_sim/third_party/checkpoints}"
G="${GRASP_ENV:-/home/felix/miniforge3/envs/grasp}/bin/gdown"

case "$WHICH" in
    rs) ID=1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk ;;   # RealSense split
    kn) ID=1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX ;;   # Kinect split
    *)  echo "usage: $0 [rs|kn]" >&2; exit 2 ;;
esac
DEST="$DEST_DIR/checkpoint-$WHICH.tar"

mkdir -p "$DEST_DIR"
if [ -s "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

echo "trying gdown..."
if "$G" "$ID" -O "$DEST" && [ -s "$DEST" ]; then
    echo "downloaded $DEST"
    sha256sum "$DEST"
    exit 0
fi

rm -f "$DEST"
cat >&2 <<EOF

------------------------------------------------------------------------------
Google Drive refused the download (per-file quota, not a local problem).

Fetch it by hand in a browser, then drop it at the path below:

    https://drive.google.com/file/d/$ID/view
    -> $DEST

Then verify with:
    ls -la $DEST
    ./serve_grasp.sh          # preflight will confirm it loads

Everything else in the pipeline is already built and does not depend on this file;
only the detector arm is blocked until it lands.
------------------------------------------------------------------------------
EOF
exit 1
