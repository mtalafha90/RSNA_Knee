#!/usr/bin/env bash
# Put the five artefacts a run needs into artefacts/, so no command line has to
# name them again.
#
#   ./scripts/setup_artefacts.sh /path/to/CNN_CPC /path/to/competition-data
#
# The four small ones are copied, so this checkout is self-contained and a later
# change in the archive cannot alter a run already under way. The dataset is
# symlinked: it is far too large to duplicate and it never changes.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${1:?usage: setup_artefacts.sh /path/to/CNN_CPC /path/to/competition-data}"
DATA="${2:?usage: setup_artefacts.sh /path/to/CNN_CPC /path/to/competition-data}"

LABELS="$ARCHIVE/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all"
BASE="$ARCHIVE/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"
POLICY="$ARCHIVE/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"
SPLIT="$ARCHIVE/runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split"

# Check everything before copying anything, so a wrong archive path does not
# leave a half-populated artefacts/ that looks set up.
fail=0
for path in "$LABELS/training_targets.csv" "$BASE" "$POLICY" "$SPLIT/b50_selection_split.json"; do
  [ -e "$path" ] || { echo "MISSING: $path"; fail=1; }
done
for name in train.csv train_series.csv; do
  [ -f "$DATA/$name" ] || { echo "MISSING: $DATA/$name"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo; echo "Nothing was copied. Is $ARCHIVE up to date? Try git pull there."; exit 1; }

mkdir -p artefacts/labels artefacts/split

for name in training_targets.csv policy.json audit.json; do
  cp "$LABELS/$name" artefacts/labels/
done
cp "$POLICY" artefacts/series_policy.json
cp "$BASE"   artefacts/base_model.pt
for name in b50_selection_split.json b50_selection_split_by_study.csv; do
  cp "$SPLIT/$name" artefacts/split/
done

# A symlink, not a copy. Replaced rather than nested if it already exists.
rm -rf artefacts/data
ln -s "$(cd "$DATA" && pwd)" artefacts/data

( cd artefacts && find . -type f -not -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )

# du does not follow symlinks by default, so this is the copied size only.
echo "artefacts/ is ready ($(du -sh artefacts | cut -f1) copied, plus the linked dataset)"
echo
echo "Check it before spending a GPU:"
echo "  rsna-knee-coverage --data-root artefacts/data"
echo "  rsna-knee-train --epochs 6 --all-data --preflight-only"
