#!/bin/bash
# ★★★ P13 停放层日常清扫（7 天规则）+ 检查点 blob 压缩。
#
#   1) 删掉 park-stream/tag-*/*/ 里 mtime 超过 KEEP_DAYS 天的会话目录
#      （那条会话的层 KV / 池化行 / tokens / 长度清单 / 草稿镜像一起走）
#   2) 压缩 checkpoints.bin：只追加的内容寻址流，删目录会在里面留下孤儿记录
#      （压缩工具自带"先验证再提交"：算不出某个存活会话的键就整体放弃）
#
#   用法：p13-park-sweep.sh [--dry-run] [--days N]
#   环境：STRATA_PARK_STREAM（默认 $HOME/Strata-data/park-stream；systemd 里建议显式写
#         Environment=STRATA_PARK_STREAM=/path/to/park-stream，别依赖跑这个服务的用户是谁）
#         P13_COMPACT_TOOL（默认 <脚本目录>/p13-blob-compact）
set -u
ROOT=${STRATA_PARK_STREAM:-$HOME/Strata-data/park-stream}
DAYS=${P13_KEEP_DAYS:-7}
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --days) shift; DAYS=${1:-7} ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL=${P13_COMPACT_TOOL:-$HERE/p13-blob-compact}

echo "[$(date '+%F %T')] P13 清扫开始：root=$ROOT 保留 $DAYS 天 dry-run=$DRY"
[ -d "$ROOT" ] || { echo "  $ROOT 不存在 ⇒ 没有可扫的停放层。如果这是配错了，给服务写 Environment=STRATA_PARK_STREAM=…" >&2; exit 0; }

# 1) 超期会话目录
before=$(du -sb "$ROOT" 2>/dev/null | cut -f1)
n=0
while IFS= read -r -d '' d; do
  sz=$(du -sb "$d" 2>/dev/null | cut -f1)
  if [ "$DRY" = "1" ]; then
    echo "  [dry-run] 会删 $d ($((sz/1048576)) MiB)"
  else
    echo "  删 $d ($((sz/1048576)) MiB)"
    rm -rf -- "$d"
  fi
  n=$((n+1))
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -type d -mtime +"$DAYS" -print0 2>/dev/null)
echo "  超期会话目录 $n 个"
after=$(du -sb "$ROOT" 2>/dev/null | cut -f1)
echo "  目录占用 $((before/1048576)) MiB → $((after/1048576)) MiB"

# 2) 压缩检查点 blob（目录删完再压，孤儿才最少）
if [ -x "$TOOL" ]; then
  if [ "$DRY" = "1" ]; then
    "$TOOL" --root "$ROOT" --days "$DAYS" --dry-run || echo "  （压缩 dry-run 放弃：见上面的原因）"
  else
    "$TOOL" --root "$ROOT" --days "$DAYS" || echo "  （压缩放弃：见上面的原因；blob 原样保留）"
  fi
else
  echo "  找不到压缩工具 $TOOL ⇒ 只清了目录（blob 里的孤儿要等工具装好再压）"
fi
echo "[$(date '+%F %T')] P13 清扫结束"
