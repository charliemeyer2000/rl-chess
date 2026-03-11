#!/bin/bash
set -e

STOCKFISH_DIR="$HOME/.local/bin"
STOCKFISH_PATH="$STOCKFISH_DIR/stockfish"

if [ -f "$STOCKFISH_PATH" ]; then
    echo "Stockfish already installed at $STOCKFISH_PATH"
    "$STOCKFISH_PATH" --help 2>&1 | head -1 || true
    exit 0
fi

mkdir -p "$STOCKFISH_DIR"

echo "Downloading Stockfish 17..."
TMPDIR=$(mktemp -d)
curl -sL "https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-avx2.tar" \
    | tar xf - -C "$TMPDIR"

BINARY=$(find "$TMPDIR" -name "stockfish*" -type f -executable | head -1)
if [ -z "$BINARY" ]; then
    BINARY=$(find "$TMPDIR" -name "stockfish*" -type f | head -1)
fi

if [ -z "$BINARY" ]; then
    echo "ERROR: Could not find stockfish binary in archive"
    ls -lR "$TMPDIR"
    exit 1
fi

cp "$BINARY" "$STOCKFISH_PATH"
chmod +x "$STOCKFISH_PATH"
rm -rf "$TMPDIR"

echo "Stockfish installed at $STOCKFISH_PATH"
"$STOCKFISH_PATH" --help 2>&1 | head -1 || true
