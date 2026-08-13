#!/usr/bin/env bash
# codex CLI 래퍼 — OCEAN 2026.
# 프롬프트를 argv 가 아니라 stdin 으로 넘긴다. PowerShell 5.1 은 긴 UTF-8 argv 를 유실시켜
# codex 가 stdin 대기에 빠진다. `.ps1` 판을 만들지 마라.
#
#   bash scripts/cx.sh P3/specs/T1-name.md        # 구현 (luna xhigh, 기본)
#   bash scripts/cx.sh -r P3/specs/question.md    # 읽기 전용
#   bash scripts/cx.sh -f P3/specs/save.md        # 네트워크 허용 (git push 전용)
#
# ⚠ -f 는 샌드박스를 완전히 해제한다. 학습·실험 사양에는 쓰지 마라.
set -euo pipefail

sandbox="workspace-write"
model="gpt-5.6-luna"
effort="xhigh"
seen_read_only=0
seen_full_access=0
if [ "${CX_MODEL+x}" = x ] && [ "$CX_MODEL" != "$model" ]; then
  echo "CX_MODEL은 $model 만 허용한다" >&2; exit 2
fi
if [ "${CX_EFFORT+x}" = x ] && [ "$CX_EFFORT" != "$effort" ]; then
  echo "CX_EFFORT는 $effort 만 허용한다" >&2; exit 2
fi
while :; do
  case "${1:-}" in
    -r) sandbox="read-only"; seen_read_only=1; shift ;;
    -s) echo "-s는 허용하지 않는다" >&2; exit 2 ;;
    -f) sandbox="danger-full-access"; seen_full_access=1; shift ;;
    *) break ;;
  esac
done
[ "$seen_read_only" -eq 1 ] && [ "$seen_full_access" -eq 1 ] && {
  echo "-r과 -f는 함께 사용할 수 없다" >&2; exit 2;
}

spec="${1:?사양 파일 경로 필요}"
[ -f "$spec" ] || { echo "사양 파일 없음: $spec" >&2; exit 1; }

root="$(cd "$(dirname "$0")/.." && pwd)"
python "$root/scripts/preflight.py" spec "$spec" || {
  echo "사양 게이트 STOP — 고치고 다시 불러라 (호출 1회 ≈ 8.7k 토큰)" >&2; exit 1; }

tree="$(printf '%s' "$spec" | sed -nE 's#^(P[123])/.*#\1#p')"

# numpy·lightgbm 은 user site-packages(AppData\Roaming)에 있다. codex 샌드박스가 APPDATA 를
# 넘겨주지 않으면 python 이 그 경로를 sys.path 에 못 넣어 ModuleNotFoundError 로 죽는다
# (2026-08-13 T4 1회 실패). 경로를 명시적으로 박는다.
env_inject=()
usersite="$(python -c 'import site;print(site.getusersitepackages())' 2>/dev/null || true)"
[ -n "$usersite" ] && export PYTHONPATH="${usersite}${PYTHONPATH:+;$PYTHONPATH}"
# codex 는 자식 셸 env 를 기본 "core" 로 스크럽해서 위 export 가 안 넘어간다.
# shell_environment_policy 로 직접 주입한다 (2026-08-13 T6·T8 2회 실패).
# 값은 TOML 로 파싱된다. Windows 경로의 백슬래시가 basic string 에서는 잘못된 이스케이프(\U)라
# 통째로 무시된다. TOML literal string(작은따옴표)을 써야 한다.
[ -n "$usersite" ] && env_inject=(-c "shell_environment_policy.set.PYTHONPATH='${usersite}'")

{
  echo "Codex CLI가 자동 로드한 루트 AGENTS.md가 최상위 규약이다. 전문을 다시 출력하지 말고 준수하라."
  echo "Windows PowerShell로 UTF-8 파일을 읽을 때는 반드시 Get-Content -Encoding UTF8을 사용하라."
  if [ -n "$tree" ] && [ -f "$root/$tree/AGENTS.md" ]; then
    echo "아래는 작업 트리의 $tree/AGENTS.md다. 이 내용도 준수하라."
    echo "--- $tree/AGENTS.md ---"
    cat "$root/$tree/AGENTS.md"
  fi
  echo "--- 이번 작업 사양 (원본: $spec) ---"
  cat "$spec"
} | codex exec \
      --model "$model" \
      -c model_reasoning_effort="$effort" \
      -c mcp_servers.fusion.enabled=false \
      "${env_inject[@]}" \
      --sandbox "$sandbox" \
      --skip-git-repo-check
