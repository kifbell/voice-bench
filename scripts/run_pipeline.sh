#!/bin/bash
# Full post-processing pipeline: parquet -> stats.json -> tables/values/figures -> report.tex
# Idempotent: each step writes deterministic outputs and can be re-run safely.
#
# Usage:
#   bash scripts/run_pipeline.sh             # default paths (uses kifbell agents_workdir)
#   PARQUET=/path/to/metrics.parquet bash scripts/run_pipeline.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PARQUET="${PARQUET:-/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/metrics.parquet}"
CALIB="${CALIB:-/mnt/gefs-me/rl/users/kifbell/diploma/voice-bench-outputs/results/calibration_anchors.json}"
RATECARDS="${RATECARDS:-/home/kifbell/sweets/twix/agents_workdir/cost_ratecards.json}"
STATS="${STATS:-/home/kifbell/sweets/twix/agents_workdir/voice-bench-stats.json}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/report}"
FIGURES_DIR="${FIGURES_DIR:-$REPORT_DIR/figures}"
TABLES_DIR="${TABLES_DIR:-$REPORT_DIR/tables}"

echo "=== voice-bench post-processing pipeline ==="
echo "  parquet:    $PARQUET"
echo "  calib:      $CALIB"
echo "  ratecards:  $RATECARDS"
echo "  stats out:  $STATS"
echo "  report dir: $REPORT_DIR"
echo ""

# Sanity checks
test -f "$PARQUET"   || { echo "ERROR: parquet not found: $PARQUET"; exit 1; }
test -f "$CALIB"     || { echo "ERROR: calibration not found: $CALIB"; exit 1; }
test -f "$RATECARDS" || { echo "ERROR: ratecards not found: $RATECARDS"; exit 1; }

mkdir -p "$REPORT_DIR" "$FIGURES_DIR" "$TABLES_DIR"

# --- Step 1: stat_pack ---
echo "[1/5] running stat_pack.py ..."
twix-python scripts/stat_pack.py \
    --parquet "$PARQUET" \
    --calibration "$CALIB" \
    --ratecards "$RATECARDS" \
    --out "$STATS"
echo "      stats -> $STATS"
echo ""

# --- Step 2: generate_values.py ---
echo "[2/5] running generate_values.py ..."
twix-python scripts/generate_values.py \
    --stats "$STATS" \
    --out "$REPORT_DIR/values.tex"
echo ""

# --- Step 3: generate_tables.py ---
echo "[3/5] running generate_tables.py ..."
twix-python scripts/generate_tables.py \
    --stats "$STATS" \
    --parquet "$PARQUET" \
    --ratecards "$RATECARDS" \
    --out-dir "$TABLES_DIR"
echo ""

# --- Step 4: make_figures.py ---
echo "[4/5] running make_figures.py ..."
twix-python scripts/make_figures.py \
    --parquet "$PARQUET" \
    --calibration "$CALIB" \
    --out "$FIGURES_DIR"
echo ""

# --- Step 5: pdflatex (optional) ---
echo "[5/5] checking pdflatex ..."
if command -v pdflatex > /dev/null 2>&1; then
    echo "      pdflatex found, compiling report ..."
    cd "$REPORT_DIR"
    # Run twice to resolve TOC and \ref{} cross-references
    for pass_no in 1 2; do
        pdflatex -interaction=nonstopmode report.tex < /dev/null > /tmp/pdflatex.log 2>&1 || {
            echo "      pdflatex pass $pass_no FAILED, see /tmp/pdflatex.log"
            tail -30 /tmp/pdflatex.log
            exit 1
        }
    done
    echo "      pdf -> $REPORT_DIR/report.pdf"
else
    echo "      pdflatex NOT installed; skipping compilation."
    echo "      To compile manually: cd $REPORT_DIR && pdflatex report.tex"
    echo "      Or upload $REPORT_DIR/ to Overleaf."
fi

echo ""
echo "=== pipeline complete ==="
echo ""
echo "Artifacts:"
echo "  $STATS"
echo "  $REPORT_DIR/values.tex"
echo "  $TABLES_DIR/*.tex"
echo "  $FIGURES_DIR/*.pdf"
echo "  $REPORT_DIR/report.tex"
if [ -f "$REPORT_DIR/report.pdf" ]; then
    echo "  $REPORT_DIR/report.pdf"
fi
