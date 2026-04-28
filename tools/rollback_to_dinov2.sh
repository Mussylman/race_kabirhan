#!/usr/bin/env bash
# rollback_to_dinov2.sh — откат SGIE color classifier с OSNet на DINOv2 fallback.
#
# Меняет только deepstream/configs/sgie_color.txt — обе функции
# (NvDsInferClassifierParseCustomDinov2 и ...ParseCustomOsnet) уже собраны
# в libnvdsinfer_racevision.so, пересборка не нужна.
#
# Usage:
#   bash tools/rollback_to_dinov2.sh
#   # затем:
#   #   ./scripts/rv.sh stop && ./scripts/rv.sh start
#   # БЕЗ RV_TIGHT_SGIE=0 — для DINOv2 default 1 (tight bbox preprocessing on).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SGIE="$REPO/deepstream/configs/sgie_color.txt"
BACKUP="$REPO/deepstream/configs/sgie_color_dinov2_backup.txt"

if [[ ! -f "$BACKUP" ]]; then
    echo "ERROR: $BACKUP not found — нечего откатывать" >&2
    exit 1
fi

# Прячем текущий OSNet config рядом, на случай повторного включения.
cp "$SGIE" "$REPO/deepstream/configs/sgie_color_osnet_backup.txt"
cp "$BACKUP" "$SGIE"

echo "OK: восстановлен DINOv2 sgie_color.txt из бэкапа."
echo "    saved current OSNet config to: deepstream/configs/sgie_color_osnet_backup.txt"
echo
echo "Перезапусти pipeline:"
echo "    ./scripts/rv.sh stop"
echo "    ./scripts/rv.sh start              # default RV_TIGHT_SGIE=1 для DINOv2"
echo
echo "Чтобы вернуться на OSNet:"
echo "    cp deepstream/configs/sgie_color_osnet_backup.txt deepstream/configs/sgie_color.txt"
echo "    RV_TIGHT_SGIE=0 ./scripts/rv.sh start"
