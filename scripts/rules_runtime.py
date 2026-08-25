"""RULES.md §0의 하드 제약을 읽는 단일 런타임 파서."""
import datetime as dt
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parent.parent
KST = dt.timezone(dt.timedelta(hours=9), name="KST")
DATE = r"(?P<year>\d{4})\s*\.\s*(?P<month>\d{1,2})\s*\.\s*(?P<day>\d{1,2})\s*\."


def _section():
    text = (ROOT / "RULES.md").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^## 0\..*?\n(.*?)(?=^##\s|\Z)", text)
    if not match:
        raise RuntimeError("RULES.md §0 없음")
    return match.group(1)


def _rows():
    rows = {}
    for line in _section().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 4 and not cells[0].startswith(":") and not set(cells[0]) <= {"-", " ", ":"}:
            rows.setdefault(cells[0], []).append(cells[1])
    return rows


def _row(rows, name):
    values = rows.get(name, [])
    if len(values) != 1:
        raise RuntimeError(f"항목 행 개수 오류: {name}={len(values)}")
    return values[0]


def _date(match):
    try:
        return dt.date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
    except (AttributeError, ValueError) as e:
        raise RuntimeError("날짜 형식 오류") from e


def _find_date(text):
    match = re.search(DATE, text)
    if not match:
        raise RuntimeError("날짜를 뽑지 못함")
    return _date(match)


def load():
    rows = _rows()
    quota_row = _row(rows, "대학부 예측 답안 형식·횟수")
    pred_row = _row(rows, "대학부 예선 제출 마감")
    model_row = _row(rows, "모델 제출")
    quota_match = re.search(r"하루\s*(?P<quota>\d+)\s*회", quota_row)
    pred_match = re.search(DATE + r"\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})", pred_row)
    if not quota_match or not pred_match:
        raise RuntimeError("규정 정규식 추출 실패")
    pred_date = _date(pred_match)
    try:
        deadline_pred = dt.datetime(
            pred_date.year, pred_date.month, pred_date.day,
            int(pred_match.group("hour")), int(pred_match.group("minute")), tzinfo=KST,
        )
        deadline_model = _find_date(model_row)
        submit_open = _find_date(quota_row)
        quota = int(quota_match.group("quota"))
    except (TypeError, ValueError) as e:
        raise RuntimeError("규정 값 형식 오류") from e
    return {
        "deadline_pred": deadline_pred,
        "deadline_model": deadline_model,
        "quota_per_problem_per_day": quota,
        "submit_open": submit_open,
    }


def _selftest():
    values = load()
    assert isinstance(values["deadline_pred"], dt.datetime)
    assert isinstance(values["deadline_model"], dt.date) and not isinstance(values["deadline_model"], dt.datetime)
    assert isinstance(values["quota_per_problem_per_day"], int)
    assert isinstance(values["submit_open"], dt.date)


def main():
    try:
        if "--selftest" in sys.argv:
            _selftest()
            print("판정: OK")
            return 0
        if "--plain" in sys.argv:
            for key, value in load().items():
                rendered = value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value
                print(f"{key}={rendered}")
            return 0
        raise RuntimeError("--plain 또는 --selftest 필요")
    except Exception as e:
        if "--selftest" in sys.argv:
            print(f"판정: STOP ({type(e).__name__}: {e})")
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
