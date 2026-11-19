"""Mini NPU Simulator.

MAC(Multiply-Accumulate) 연산으로 입력 패턴이 Cross 필터와 X 필터 중
어디에 더 가까운지 판정한다. 표준 라이브러리만 사용한다.

실행: python main.py
"""

import json
import math
import os
import sys
import time

EPSILON = 1e-9
REPEAT = 10
PERF_SIZES = (3, 5, 13, 25)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 원본 표기 -> 표준 라벨 (expected 값과 filter 키 모두 여기서 처리)
LABEL_MAP = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

LINE = "#" + "-" * 39


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
class Matrix:
    """N×N 2차원 숫자 배열."""

    def __init__(self, size, rows=None):
        self.size = size
        if rows is None:
            self.data = [[0.0] * size for _ in range(size)]
        else:
            self.data = [[float(v) for v in row] for row in rows]

    def get(self, r, c):
        return self.data[r][c]

    def set(self, r, c, value):
        self.data[r][c] = float(value)

    def flatten(self):
        """보너스: 길이 N²의 1차원 리스트로 변환."""
        flat = []
        for row in self.data:
            for v in row:
                flat.append(v)
        return flat

    @classmethod
    def from_rows(cls, rows):
        """2차원 리스트를 검증 후 Matrix로 만든다. 정사각형이 아니면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열이 아니거나 비어 있음")
        n = len(rows)
        for i, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{i + 1}번째 행이 배열이 아님")
            if len(row) != n:
                raise ValueError(f"{i + 1}번째 행의 열 수 {len(row)} != 행 수 {n}")
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"{i + 1}번째 행에 숫자가 아닌 값: {v!r}")
                to_finite_float(v, f"{i + 1}번째 행")
        return cls(n, rows)


def to_finite_float(value, where="값"):
    """유한한 float로 변환. 변환 불가·너무 큰 수·nan/inf면 ValueError."""
    try:
        f = float(value)
    except OverflowError:
        raise ValueError(f"{where}에 float 범위를 넘는 값") from None
    if not math.isfinite(f):
        raise ValueError(f"{where}에 유한하지 않은 값(nan/inf): {value!r}")
    return f


# ---------------------------------------------------------------------------
# 핵심 로직
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """위치별 곱을 모두 더한다. 반복문으로 직접 구현 (N² 회 곱셈)."""
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, "
                         f"필터 {filt.size}×{filt.size}")
    n = pattern.size
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern.get(i, j) * filt.get(i, j)
    return total


def mac_flat(pattern_flat, filter_flat):
    """보너스: 1차원 배열 기반 MAC."""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


def normalize_label(raw):
    """'+', 'cross' -> Cross / 'x' -> X. 알 수 없으면 ValueError."""
    key = str(raw).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"알 수 없는 라벨: {raw!r}")
    return LABEL_MAP[key]


def judge(score_a, score_b, label_a, label_b, eps=EPSILON):
    """epsilon 기반 비교. 차이가 eps 미만이면 None(동점).

    곱셈 누적이 overflow되어 점수가 inf/nan이 되면 비교 근거가 없으므로 None.
    """
    if not (math.isfinite(score_a) and math.isfinite(score_b)):
        return None
    if abs(score_a - score_b) < eps:
        return None
    return label_a if score_a > score_b else label_b


def measure_ms(func, *args, repeat=REPEAT):
    """func(*args)를 repeat번 호출한 평균 시간(ms). 연산 구간만 측정."""
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


# ---------------------------------------------------------------------------
# 보너스: 패턴 생성기
# ---------------------------------------------------------------------------
def make_cross(n):
    m = Matrix(n)
    c = n // 2
    for i in range(n):
        m.set(c, i, 1)
        m.set(i, c, 1)
    return m


def make_x(n):
    m = Matrix(n)
    for i in range(n):
        m.set(i, i, 1)
        m.set(i, n - 1 - i, 1)
    return m


# ---------------------------------------------------------------------------
# 출력 도우미
# ---------------------------------------------------------------------------
def section(title):
    print()
    print(LINE)
    print(f"# {title}")
    print(LINE)


def print_matrix(m):
    for row in m.data:
        print("  " + " ".join(f"{v:g}" for v in row))


def print_perf_table(sizes):
    """크기별 평균 시간과 연산 횟수 표. 보너스로 1차원 방식도 함께 측정."""
    print(f"{'크기':<8}{'평균 시간(ms)':>14}{'1D 평균(ms)':>14}{'연산 횟수(N²)':>16}")
    print("-" * 54)
    for n in sizes:
        pattern, filt = make_cross(n), make_x(n)
        ms_2d = measure_ms(mac, pattern, filt)
        p_flat, f_flat = pattern.flatten(), filt.flatten()
        ms_1d = measure_ms(mac_flat, p_flat, f_flat)
        label = f"{n}×{n}"
        print(f"{label:<8}{ms_2d:>14.4f}{ms_1d:>14.4f}{n * n:>16}")
    print(f"(각 크기별 {REPEAT}회 반복 평균, I/O 제외, 1D = 1차원 배열 최적화 버전)")


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
GENERATORS = {"cross": make_cross, "x": make_x}


def read_matrix(title, n=3):
    """n줄을 입력받는다. 형식 오류면 안내 후 처음부터 다시 입력받는다.

    첫 줄에 'cross' 또는 'x'만 입력하면 패턴 생성기로 n×n 행렬을 만든다.
    """
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분 | 'cross' 또는 'x' 입력 시 자동 생성)")
        rows = []
        error = None
        for line_no in range(n):
            line = input()
            if line_no == 0 and line.strip().lower() in GENERATORS:
                m = GENERATORS[line.strip().lower()](n)
                print(f"→ {normalize_label(line)} 패턴 자동 생성:")
                print_matrix(m)
                return m
            parts = line.split()
            if len(parts) != n:
                error = f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                error = "입력 형식 오류: 숫자만 입력하세요."
                break
            if not all(math.isfinite(v) for v in rows[-1]):
                error = "입력 형식 오류: nan, inf는 사용할 수 없습니다. 유한한 숫자를 입력하세요."
                break
        if error is None:
            return Matrix(n, rows)
        print(error)
        print(f"→ [{title}] 처음부터 다시 입력하세요.\n")


def run_user_mode():
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")

    print("\n✓ 필터 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed = measure_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    result = judge(score_a, score_b, "A", "B")

    section("[3] MAC 결과")
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간(평균/{REPEAT}회): {elapsed:.4f} ms")
    if result is None and not (math.isfinite(score_a) and math.isfinite(score_b)):
        print("판정: 판정 불가 (점수 overflow: inf/nan)")
    elif result is None:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})")
    else:
        print(f"판정: {result}")

    section(f"[4] 성능 분석 (평균/{REPEAT}회)")
    print_perf_table((3,))


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def load_filters(raw_filters):
    """{N: {Cross: Matrix, X: Matrix}} 반환. 문제 있는 필터는 건너뛰고 경고."""
    filters = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 형식이 잘못됨")
        return filters
    for key, group in raw_filters.items():
        try:
            if not key.startswith("size_"):
                raise ValueError("키가 size_N 형식이 아님")
            n = int(key[len("size_"):])
            if not isinstance(group, dict):
                raise ValueError("필터 묶음이 객체가 아님")
            loaded = {}
            for label_key, rows in group.items():
                label = normalize_label(label_key)
                m = Matrix.from_rows(rows)
                if m.size != n:
                    raise ValueError(f"{label_key} 필터 크기 {m.size} != {n}")
                loaded[label] = m
            missing = [lb for lb in (LABEL_CROSS, LABEL_X) if lb not in loaded]
            if missing:
                raise ValueError(f"필터 누락: {', '.join(missing)}")
            filters[n] = loaded
            print(f"✓ {key:<8} 필터 로드 완료 ({LABEL_CROSS}, {LABEL_X})")
        except (ValueError, TypeError, AttributeError, OverflowError) as e:
            print(f"✗ {key} 필터 로드 실패: {e}")
    return filters


def size_from_key(key):
    """'size_13_2' -> 13."""
    parts = str(key).split("_")
    if len(parts) != 3 or parts[0] != "size":
        raise ValueError(f"키 형식 오류(size_N_idx 아님): {key}")
    return int(parts[1])


def analyze_case(key, case, filters):
    """한 케이스를 분석해 (통과 여부, 실패 사유) 반환. 예외는 밖으로 내보내지 않는다."""
    print(f"--- {key} ---")
    try:
        if not isinstance(case, dict):
            raise ValueError("케이스가 객체가 아님")
        if "input" not in case or "expected" not in case:
            raise ValueError("input 또는 expected 누락")

        n = size_from_key(key)
        if n not in filters:
            raise ValueError(f"size_{n} 필터 없음")
        expected = normalize_label(case["expected"])
        pattern = Matrix.from_rows(case["input"])
        if pattern.size != n:
            raise ValueError(f"크기 불일치: 키는 {n}×{n}, 패턴은 {pattern.size}×{pattern.size}")

        cross_score = mac(pattern, filters[n][LABEL_CROSS])
        x_score = mac(pattern, filters[n][LABEL_X])
        decision = judge(cross_score, x_score, LABEL_CROSS, LABEL_X) or LABEL_UNDECIDED
    except (ValueError, TypeError, OverflowError) as e:
        print(f"FAIL (스키마/데이터 오류: {e})")
        return False, f"스키마/데이터 오류 - {e}"

    print(f"Cross 점수: {cross_score!r}")
    print(f"X 점수: {x_score!r}")
    passed = decision == expected
    status = "PASS" if passed else "FAIL"
    note = " (동점 규칙)" if decision == LABEL_UNDECIDED else ""
    print(f"판정: {decision} | expected: {expected} | {status}{note}")

    if passed:
        return True, None
    if decision == LABEL_UNDECIDED and not (math.isfinite(cross_score) and math.isfinite(x_score)):
        return False, f"점수 overflow(inf/nan)로 판정 불가 (Cross={cross_score!r}, X={x_score!r})"
    if decision == LABEL_UNDECIDED:
        return False, (f"동점(UNDECIDED) 처리 규칙에 따라 FAIL "
                       f"(Cross={cross_score!r}, X={x_score!r}, |차이| < {EPSILON:g})")
    return False, f"판정 {decision} != expected {expected}"


def run_json_mode(path=DATA_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {path}")
        return
    except json.JSONDecodeError as e:
        print(f"✗ JSON 파싱 실패: {e}")
        return
    if not isinstance(data, dict):
        print("✗ data.json 최상위가 객체가 아닙니다.")
        return

    section("[1] 필터 로드")
    filters = load_filters(data.get("filters"))

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        print("✗ patterns 항목이 없거나 형식이 잘못됨")
        patterns = {}

    failures = []
    passed_count = 0
    for key, case in patterns.items():
        ok, reason = analyze_case(key, case, filters)
        if ok:
            passed_count += 1
        else:
            failures.append((key, reason))

    section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    print_perf_table(PERF_SIZES)

    section("[4] 결과 요약")
    total = len(patterns)
    print(f"총 테스트: {total}개")
    print(f"통과: {passed_count}개")
    print(f"실패: {len(failures)}개")
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print(f"- {key}: {reason}")
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션 참고)')


# ---------------------------------------------------------------------------
# 모드 3 (보너스): 패턴 생성기
# ---------------------------------------------------------------------------
def read_size():
    """2 이상의 정수 N을 입력받는다. 잘못된 값이면 재입력."""
    while True:
        raw = input("크기 N 입력 (2 이상 정수): ").strip()
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n >= 2:
            if n % 2 == 0:
                print(f"※ N이 짝수면 정중앙이 없어 Cross는 {n // 2}번째 행/열(0부터)을 기준으로 생성됩니다.")
            return n
        print("입력 형식 오류: 2 이상의 정수를 입력하세요.")


def run_generator_mode():
    section("[1] 크기 입력")
    n = read_size()
    cross, x = make_cross(n), make_x(n)

    section(f"[2] 생성된 패턴 ({n}×{n})")
    if n <= 13:
        print("Cross:")
        print_matrix(cross)
        print("X:")
        print_matrix(x)
    else:
        print(f"(N > 13은 출력 생략) Cross 1의 개수: {mac(cross, cross):g}, X 1의 개수: {mac(x, x):g}")

    section("[3] MAC 결과 (생성 패턴 × 생성 필터)")
    for name, pattern in ((LABEL_CROSS, cross), (LABEL_X, x)):
        cs, xs = mac(pattern, cross), mac(pattern, x)
        decision = judge(cs, xs, LABEL_CROSS, LABEL_X) or LABEL_UNDECIDED
        status = "PASS" if decision == name else "FAIL"
        print(f"{name} 패턴 → Cross 점수: {cs!r}, X 점수: {xs!r}, 판정: {decision} | {status}")

    section(f"[4] 성능 분석 (평균/{REPEAT}회)")
    print_perf_table((n,))


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main():
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 패턴 생성기 (N×N, 보너스)")
    modes = {"1": run_user_mode, "2": run_json_mode, "3": run_generator_mode}
    while True:
        choice = input("선택: ").strip()
        if choice in modes:
            modes[choice]()
            return
        print("1, 2, 3 중 하나를 입력하세요.")


if __name__ == "__main__":
    # Windows(cp949)에서 출력이 파이프/파일로 리다이렉트될 때 ✓ 같은 문자로 죽지 않도록
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        sys.exit(0)
