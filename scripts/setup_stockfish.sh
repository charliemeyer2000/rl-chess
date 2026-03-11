#!/bin/bash
set -e

STOCKFISH_DIR="$HOME/.local/bin"
STOCKFISH_PATH="$STOCKFISH_DIR/stockfish"

if [ -f "$STOCKFISH_PATH" ]; then
    echo "Stockfish already installed at $STOCKFISH_PATH"
    "$STOCKFISH_PATH" <<< "quit" 2>&1 | head -1 || true
    exit 0
fi

mkdir -p "$STOCKFISH_DIR"

module load gcc/14.2.0 2>/dev/null || true

echo "Downloading Stockfish 17 source..."
TMPDIR=$(mktemp -d)
curl -sL "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_17.tar.gz" \
    | tar xzf - -C "$TMPDIR"

cd "$TMPDIR/Stockfish-sf_17/src"
echo "Compiling Stockfish (this takes ~2 minutes)..."
make -j$(nproc) build ARCH=x86-64-avx2 \
    EXTRACXXFLAGS="-static-libstdc++ -static-libgcc" \
    EXTRALDFLAGS="-static-libstdc++ -static-libgcc" 2>&1 | tail -5

cp stockfish "$STOCKFISH_PATH"
chmod +x "$STOCKFISH_PATH"
rm -rf "$TMPDIR"

echo "Stockfish installed at $STOCKFISH_PATH"
"$STOCKFISH_PATH" <<< "quit" 2>&1 | head -1 || true
