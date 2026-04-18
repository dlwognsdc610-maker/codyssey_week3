"""
Mini NPU Simulator
==================
MAC(Multiply-Accumulate) 연산 기반 패턴 판별 시뮬레이터
외부 라이브러리 사용 금지 - 표준 라이브러리(json, time)만 사용
"""

import json
import time

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
EPSILON = 1e-9          # 동점 판정 허용 오차
REPEAT  = 10            # 성능 측정 반복 횟수

# 라벨 정규화 맵
LABEL_MAP = {
    "+":     "Cross",
    "cross": "Cross",
    "Cross": "Cross",
    "x":     "X",
    "X":     "X",
}


# ─────────────────────────────────────────────
# 1. 데이터 구조 : Pattern2D
# ─────────────────────────────────────────────
class Pattern2D:
    """N×N 크기의 2차원 숫자 패턴을 저장하고 접근하는 자료구조."""

    def __init__(self, data: list[list[float]]):
        self._data = data
        self.size  = len(data)

    def get(self, row: int, col: int) -> float:
        return self._data[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        self._data[row][col] = value

    def rows(self) -> int:
        return self.size

    def cols(self) -> int:
        return len(self._data[0]) if self._data else 0

    def is_square(self) -> bool:
        return self.rows() == self.cols()

    def __repr__(self) -> str:
        lines = []
        for row in self._data:
            lines.append("  " + "  ".join(f"{v:4.1f}" for v in row))
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 2. 라벨 정규화
# ─────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    """원시 라벨을 표준 라벨(Cross / X)로 변환한다."""
    key = raw.strip().lower() if raw.strip() not in ("+",) else raw.strip()
    # '+' 는 소문자화하면 '+' 그대로 → LABEL_MAP 에서 직접 매핑
    result = LABEL_MAP.get(raw.strip()) or LABEL_MAP.get(key)
    if result is None:
        raise ValueError(f"알 수 없는 라벨: '{raw}'")
    return result


# ─────────────────────────────────────────────
# 3. MAC 연산 (순수 반복문 구현)
# ─────────────────────────────────────────────
def mac(pattern: Pattern2D, filt: Pattern2D) -> float:
    """
    입력 패턴과 필터의 MAC(Multiply-Accumulate) 연산.
    두 배열의 같은 위치 값을 곱하고 모두 더해 유사도 점수를 반환한다.
    """
    n      = pattern.size
    total  = 0.0
    for r in range(n):
        for c in range(n):
            total += pattern.get(r, c) * filt.get(r, c)
    return total


# ─────────────────────────────────────────────
# 4. 판정 로직
# ─────────────────────────────────────────────
def judge(score_a: float, score_b: float,
          label_a: str = "Cross", label_b: str = "X") -> str:
    """
    두 필터 점수를 비교하여 판정 결과(label_a / label_b / UNDECIDED)를 반환.
    |score_a - score_b| < EPSILON 이면 동점(UNDECIDED).
    """
    diff = score_a - score_b
    if abs(diff) < EPSILON:
        return "UNDECIDED"
    return label_a if diff > 0 else label_b


# ─────────────────────────────────────────────
# 5. 성능 측정
# ─────────────────────────────────────────────
def measure_mac_time(pattern: Pattern2D, filt: Pattern2D,
                     repeat: int = REPEAT) -> float:
    """MAC 연산을 repeat 회 반복 측정 후 평균 시간(ms)을 반환.
    I/O 시간을 제외하고 연산 구간만 측정한다."""
    total_ns = 0
    for _ in range(repeat):
        t0        = time.perf_counter_ns()
        mac(pattern, filt)
        total_ns += time.perf_counter_ns() - t0
    return (total_ns / repeat) / 1_000_000   # ns → ms


def print_performance_table(entries: list[tuple[int, float]]) -> None:
    """크기(N×N) / 평균 시간(ms) / 연산 횟수(N²) 표를 출력한다."""
    sep = "+" + "-"*10 + "+" + "-"*18 + "+" + "-"*14 + "+"
    print(sep)
    print(f"| {'N×N':^8} | {'평균 시간(ms)':^16} | {'연산 횟수(N²)':^12} |")
    print(sep)
    for n, avg_ms in entries:
        ops = n * n
        print(f"| {f'{n}×{n}':^8} | {avg_ms:^16.6f} | {ops:^12,} |")
    print(sep)


# ─────────────────────────────────────────────
# 6. 콘솔 입력 헬퍼
# ─────────────────────────────────────────────
def input_matrix(name: str, n: int) -> Pattern2D:
    """콘솔에서 n×n 행렬을 입력받아 Pattern2D 로 반환한다.
    행/열 개수 불일치, 숫자 파싱 실패 시 안내 후 재입력을 유도한다."""
    print(f"\n[{name}] {n}×{n} 행렬을 한 줄씩 입력하세요 (공백으로 숫자 구분):")
    rows = []
    while len(rows) < n:
        line_num = len(rows) + 1
        raw = input(f"  행 {line_num}: ").strip()
        try:
            values = [float(v) for v in raw.split()]
        except ValueError:
            print(f"  ※ 입력 형식 오류: 숫자만 입력 가능합니다. 다시 입력하세요.")
            continue
        if len(values) != n:
            print(f"  ※ 입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
            continue
        rows.append(values)
    return Pattern2D(rows)


def print_matrix(p: Pattern2D, label: str = "") -> None:
    if label:
        print(f"  [{label}]")
    print(repr(p))


# ─────────────────────────────────────────────
# 7. 모드 1 : 사용자 직접 입력 (3×3)
# ─────────────────────────────────────────────
def mode_manual() -> list[tuple[int, float]]:
    """
    사용자가 3×3 필터 2개(A=Cross, B=X)와 패턴을 직접 입력한다.
    MAC 점수, 연산 시간, 판정 결과를 출력하고
    성능 측정 결과(튜플 리스트)를 반환한다.
    """
    N = 3
    print("\n" + "="*55)
    print("  모드 1 : 사용자 입력 (3×3)")
    print("="*55)

    filter_a = input_matrix("필터 A (Cross)", N)
    print("\n  ✓ 필터 A 저장 완료:")
    print_matrix(filter_a, "Cross Filter")

    filter_b = input_matrix("필터 B (X)", N)
    print("\n  ✓ 필터 B 저장 완료:")
    print_matrix(filter_b, "X Filter")

    pattern = input_matrix("입력 패턴", N)
    print("\n  ✓ 패턴 저장 완료:")
    print_matrix(pattern, "Pattern")

    # MAC 연산 및 시간 측정
    t0      = time.perf_counter_ns()
    score_a = mac(pattern, filter_a)
    t1      = time.perf_counter_ns()
    score_b = mac(pattern, filter_b)
    t2      = time.perf_counter_ns()

    time_a  = (t1 - t0) / 1_000_000
    time_b  = (t2 - t1) / 1_000_000

    verdict = judge(score_a, score_b)

    print("\n" + "-"*55)
    print("  [ MAC 연산 결과 ]")
    print(f"  Cross 필터 점수 : {score_a:.4f}   (연산 시간: {time_a:.6f} ms)")
    print(f"  X     필터 점수 : {score_b:.4f}   (연산 시간: {time_b:.6f} ms)")
    print(f"  판정             : {verdict}")
    print("-"*55)

    # 성능 분석 (3×3, 10회 반복)
    avg_ms  = measure_mac_time(pattern, filter_a)
    entries = [(N, avg_ms)]
    print("\n  [ 성능 분석 ]")
    print_performance_table(entries)

    return entries


# ─────────────────────────────────────────────
# 8. JSON 로드 및 스키마 검증
# ─────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_get_filter(filters: dict, size: int,
                             case_key: str) -> tuple[Pattern2D | None,
                                                     Pattern2D | None,
                                                     str | None]:
    """
    filters 딕셔너리에서 해당 size 의 Cross/X 필터를 꺼내 Pattern2D 로 변환.
    스키마 오류 시 (None, None, error_msg) 반환.
    """
    key = f"size_{size}"
    if key not in filters:
        return None, None, f"filters.{key} 키 없음"

    block = filters[key]

    # Cross 필터 추출 (라벨 정규화)
    cross_raw = None
    x_raw     = None
    for k, v in block.items():
        try:
            std = normalize_label(k)
        except ValueError:
            continue
        if std == "Cross":
            cross_raw = v
        elif std == "X":
            x_raw = v

    if cross_raw is None:
        return None, None, f"filters.{key} 에 Cross 필터 없음"
    if x_raw is None:
        return None, None, f"filters.{key} 에 X 필터 없음"

    f_cross = Pattern2D(cross_raw)
    f_x     = Pattern2D(x_raw)

    # 크기 검증
    for lbl, flt in (("Cross", f_cross), ("X", f_x)):
        if not flt.is_square() or flt.size != size:
            return None, None, (
                f"filters.{key}.{lbl}: 크기 불일치 "
                f"(expected {size}×{size}, got {flt.rows()}×{flt.cols()})"
            )

    return f_cross, f_x, None


# ─────────────────────────────────────────────
# 9. 모드 2 : data.json 분석
# ─────────────────────────────────────────────
def mode_json(path: str = "data.json") -> list[tuple[int, float]]:
    """
    data.json 을 로드하여 모든 패턴을 판정하고 PASS/FAIL 을 출력한다.
    성능 측정 결과(튜플 리스트)를 반환한다.
    """
    print("\n" + "="*55)
    print(f"  모드 2 : data.json 분석  ({path})")
    print("="*55)

    try:
        data = load_json(path)
    except FileNotFoundError:
        print(f"  ✗ 파일을 찾을 수 없습니다: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON 파싱 오류: {e}")
        return []

    filters  = data.get("filters", {})
    patterns = data.get("patterns", {})

    total   = 0
    passed  = 0
    failed  = 0
    fail_log: list[tuple[str, str]] = []

    # 성능 측정용: 크기별 첫 번째 유효 패턴 보관
    perf_samples: dict[int, tuple[Pattern2D, Pattern2D]] = {}

    print()
    for case_key, case_val in patterns.items():
        total += 1

        # ── 크기 추출 (size_{N}_{idx} 규칙) ──────────────────────────
        parts = case_key.split("_")
        # 형식: size_<N>_<idx>  → parts = ["size", N, idx]
        if len(parts) < 3 or parts[0] != "size":
            reason = f"키 형식 오류: '{case_key}'"
            _record_fail(case_key, reason, fail_log)
            failed += 1
            continue

        try:
            n = int(parts[1])
        except ValueError:
            reason = f"크기 파싱 실패: '{parts[1]}'"
            _record_fail(case_key, reason, fail_log)
            failed += 1
            continue

        # ── 입력 패턴 구성 ────────────────────────────────────────────
        if "input" not in case_val:
            reason = "input 키 없음"
            _record_fail(case_key, reason, fail_log)
            failed += 1
            continue

        pattern = Pattern2D(case_val["input"])
        if pattern.rows() != n or pattern.cols() != n:
            reason = (
                f"패턴 크기 불일치 "
                f"(expected {n}×{n}, got {pattern.rows()}×{pattern.cols()})"
            )
            _record_fail(case_key, reason, fail_log)
            failed += 1
            continue

        # ── 필터 로드 및 검증 ─────────────────────────────────────────
        f_cross, f_x, err = validate_and_get_filter(filters, n, case_key)
        if err:
            _record_fail(case_key, err, fail_log)
            failed += 1
            continue

        # ── expected 정규화 ───────────────────────────────────────────
        raw_expected = case_val.get("expected", "")
        try:
            expected = normalize_label(str(raw_expected))
        except ValueError:
            reason = f"expected 라벨 파싱 실패: '{raw_expected}'"
            _record_fail(case_key, reason, fail_log)
            failed += 1
            continue

        # ── MAC 연산 ──────────────────────────────────────────────────
        score_cross = mac(pattern, f_cross)
        score_x     = mac(pattern, f_x)
        verdict     = judge(score_cross, score_x)
        result      = "PASS" if verdict == expected else "FAIL"

        if result == "PASS":
            passed += 1
        else:
            failed += 1
            fail_log.append((case_key, f"판정={verdict}, expected={expected}"))

        # ── 출력 ──────────────────────────────────────────────────────
        print(
            f"  {case_key:<16} | "
            f"Cross={score_cross:7.2f}  X={score_x:7.2f} | "
            f"판정={verdict:<9} | expected={expected:<5} | {result}"
        )

        # 성능 샘플 저장 (크기별 첫 번째)
        if n not in perf_samples:
            perf_samples[n] = (pattern, f_cross)

    # ── 결과 요약 ─────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print(f"  총 테스트: {total}  |  통과: {passed}  |  실패: {failed}")
    if fail_log:
        print("\n  [ 실패 케이스 목록 ]")
        for fk, fr in fail_log:
            print(f"    - {fk}: {fr}")
    print("─"*55)

    # ── 성능 분석 ─────────────────────────────────────────────────────
    # 3×3 는 직접 생성 (JSON 에 없을 수 있으므로)
    cross_3 = Pattern2D([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    pat_3 = Pattern2D([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    perf_samples.setdefault(3, (pat_3, cross_3))

    entries = []
    for n in sorted(perf_samples.keys()):
        pat, flt = perf_samples[n]
        avg_ms   = measure_mac_time(pat, flt)
        entries.append((n, avg_ms))

    print("\n  [ 성능 분석 ]")
    print_performance_table(entries)

    return entries


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────
def _record_fail(key: str, reason: str,
                 fail_log: list[tuple[str, str]]) -> None:
    fail_log.append((key, reason))
    print(f"  {key:<16} | FAIL  ({reason})")


# ─────────────────────────────────────────────
# 10. 메인 진입점
# ─────────────────────────────────────────────
def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║          Mini NPU Simulator  (MAC 연산 기반)         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  1. 사용자 직접 입력 (3×3)                           ║")
    print("║  2. data.json 파일 분석                               ║")
    print("╚══════════════════════════════════════════════════════╝")

    while True:
        choice = input("\n모드 선택 (1 또는 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ※ 1 또는 2 를 입력하세요.")

    if choice == "1":
        mode_manual()
    else:
        json_path = input("data.json 경로 (Enter = ./data.json): ").strip()
        if not json_path:
            json_path = "data.json"
        mode_json(json_path)

    print("\n  프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
