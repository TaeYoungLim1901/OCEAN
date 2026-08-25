#!/usr/bin/env bash
# grok CLI 래퍼 — OCEAN 2026.
# 역할 분담: 구현은 codex luna xhigh (`cx.cmd`). grok 은 **읽기 전용 2차 의견·조사**다.
# grok 의 고유 가치는 웹 검색이다 (codex 샌드박스에는 없다). 같은 파일을 둘이 쓰면 충돌한다.
#
#   bash scripts/gk.sh P1/specs/Q1-name.md        # 읽기 전용 + 웹 검색 (기본)
#   bash scripts/gk.sh -w P1/specs/T9-name.md     # 쓰기 허용 (예외. 사용자 승인 필요)
#
# 프롬프트는 argv 가 아니라 --prompt-file 로 넘긴다. 긴 UTF-8 argv 유실을 피한다.
# grok 은 루트 AGENTS.md·CLAUDE.md 를 자동 로드한다 (`grok inspect` 로 확인). 다시 넣지 않는다.
set -euo pipefail

model="${GK_MODEL:-grok-4.6}"
mode="read"
while :; do
  case "${1:-}" in
    -w) mode="write"; shift ;;
    *) break ;;
  esac
done

spec="${1:?사양 파일 경로 필요}"
[ -f "$spec" ] || { echo "사양 파일 없음: $spec" >&2; exit 1; }

root="$(cd "$(dirname "$0")/.." && pwd)"
python "$root/scripts/preflight.py" spec "$spec" || {
  echo "사양 게이트 STOP — 고치고 다시 불러라" >&2; exit 1; }

tree="$(printf '%s' "$spec" | sed -nE 's#^(P[123])/.*#\1#p')"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
{
  echo "자동 로드된 루트 AGENTS.md 가 최상위 규약이다. 전문을 다시 출력하지 말고 준수하라."
  if [ "$mode" = read ]; then
    echo "이 호출은 **읽기 전용**이다. 파일을 만들거나 고치거나 지우지 마라. 명령 실행도 하지 마라."
    echo "조사·분석·근거 제시만 한다. 구현이 필요하면 '구현 필요' 라고만 적고 멈춰라."
  fi
  if [ -n "$tree" ] && [ -f "$root/$tree/AGENTS.md" ]; then
    echo "아래는 작업 트리의 $tree/AGENTS.md 다. 이 내용도 준수하라."
    echo "--- $tree/AGENTS.md ---"
    cat "$root/$tree/AGENTS.md"
  fi
  echo "--- 이번 작업 사양 (원본: $spec) ---"
  cat "$spec"
} > "$tmp"

args=(-m "$model" --prompt-file "$tmp" --output-format plain)
if [ "$mode" = read ]; then
  args+=(--permission-mode plan --no-subagents)
else
  args+=(--permission-mode acceptEdits --no-subagents)
fi

grok "${args[@]}"
