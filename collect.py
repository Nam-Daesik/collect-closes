import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import yfinance as yf


# ============================================================
# 기본 설정
# ============================================================

START_DATE = "2010-01-01"
TICKERS = [
    "QQQ",
    "TQQQ",
    "SOXL",
    "TECL",
    "SGOV",
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# GitHub Actions가 서머타임 기준 UTC 시각으로 고정되어 있다면
# 미국 표준시 기간에는 기존 코드처럼 한 시간 기다린다.
# 로컬 테스트 등에서 기다리지 않으려면 환경변수를 0으로 둔다.
# STANDARD_TIME_DELAY_SECONDS=0
STANDARD_TIME_DELAY_SECONDS = int(
    os.environ.get(
        "STANDARD_TIME_DELAY_SECONDS",
        "3600",
    )
)

RECOVERY_PERIOD = "3mo"
RECOVERY_PADDING_DAYS = 7
FUTURE_TRADING_DAYS = 50
CALENDAR_HORIZON_DAYS = 120


# ============================================================
# 공통 로그
# ============================================================

def log(message):
    print(message, flush=True)


def log_warning(message):
    log(f"WARNING: {message}")


# ============================================================
# Yahoo 응답에서 단일 가격 필드 추출
#
# yfinance 버전에 따라 단일 티커도 MultiIndex DataFrame으로
# 반환될 수 있으므로 Series로 강제 정규화한다.
# ============================================================

def normalize_price_field(
    frame,
    ticker,
    field="Close",
):
    if frame is None or frame.empty:
        raise ValueError(
            f"{ticker}: empty Yahoo response"
        )

    work = frame.copy()

    if isinstance(work.columns, pd.MultiIndex):
        level_0 = work.columns.get_level_values(0)
        level_last = work.columns.get_level_values(-1)

        if field in level_0:
            selected = work.xs(
                field,
                axis=1,
                level=0,
                drop_level=True,
            )

        elif field in level_last:
            selected = work.xs(
                field,
                axis=1,
                level=-1,
                drop_level=True,
            )

        else:
            raise ValueError(
                f"{ticker}: {field} field missing"
            )

    else:
        if field not in work.columns:
            raise ValueError(
                f"{ticker}: {field} field missing"
            )

        selected = work[field]

    if isinstance(selected, pd.DataFrame):
        if ticker in selected.columns:
            series = selected[ticker]

        elif len(selected.columns) == 1:
            series = selected.iloc[:, 0]

        else:
            raise ValueError(
                f"{ticker}: ambiguous {field} columns"
            )

    else:
        series = selected

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    index = pd.DatetimeIndex(
        pd.to_datetime(series.index)
    )

    if index.tz is not None:
        index = index.tz_localize(None)

    series.index = index.normalize()
    series = series[
        ~series.index.duplicated(
            keep="last"
        )
    ]

    series = series[
        series.notna()
        & np.isfinite(series)
        & (series > 0)
    ].sort_index()

    series.name = ticker

    if series.empty:
        raise ValueError(
            f"{ticker}: no valid {field} values"
        )

    return series


# ============================================================
# Yahoo 기본 전체 수집
# ============================================================

def fetch_max_close(ticker):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False,
                ignore_tz=True,
                progress=False,
                rounding=False,
                threads=False,
            )

            return normalize_price_field(
                raw,
                ticker,
                "Close",
            )

        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES - 1:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"{ticker}: period=max fetch failed: "
        f"{last_error}"
    )


# ============================================================
# 결측 거래일 원시 종가 재수집
#
# period=max에 중간 날짜가 빠지면 다음 경로를 순차 시도한다.
#
# 1. yf.download(start/end)
# 2. yf.download(period=3mo)
# 3. Ticker.history(repair=True)
# 4. Ticker.history 일반 조회
#
# 어느 경로에서도 값이 없으면 결측을 그대로 반환한다.
# ============================================================

def recover_missing_close(
    ticker,
    base_series,
    required_days,
):
    result = base_series.copy()

    required = (
        pd.DatetimeIndex(required_days)
        .tz_localize(None)
        .normalize()
        .sort_values()
        .unique()
    )

    def find_missing():
        return required.difference(
            result.index
        )

    missing = find_missing()

    if len(missing) == 0:
        return result, missing, "MAX_COMPLETE"

    start_date = (
        pd.Timestamp(missing.min())
        - pd.Timedelta(
            days=RECOVERY_PADDING_DAYS
        )
    )

    # yfinance의 end는 exclusive이다.
    end_date = (
        pd.Timestamp(missing.max())
        + pd.Timedelta(
            days=RECOVERY_PADDING_DAYS + 1
        )
    )

    def download_range():
        return yf.download(
            ticker,
            start=start_date.strftime(
                "%Y-%m-%d"
            ),
            end=end_date.strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=False,
            actions=False,
            ignore_tz=True,
            progress=False,
            rounding=False,
            threads=False,
        )

    def download_recent():
        return yf.download(
            ticker,
            period=RECOVERY_PERIOD,
            interval="1d",
            auto_adjust=False,
            actions=False,
            ignore_tz=True,
            progress=False,
            rounding=False,
            threads=False,
        )

    def history_range_repair():
        return yf.Ticker(
            ticker
        ).history(
            start=start_date.strftime(
                "%Y-%m-%d"
            ),
            end=end_date.strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )

    def history_range_plain():
        return yf.Ticker(
            ticker
        ).history(
            start=start_date.strftime(
                "%Y-%m-%d"
            ),
            end=end_date.strftime(
                "%Y-%m-%d"
            ),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )

    routes = [
        (
            "DOWNLOAD_RANGE",
            download_range,
        ),
        (
            "DOWNLOAD_RECENT",
            download_recent,
        ),
        (
            "HISTORY_RANGE_REPAIR",
            history_range_repair,
        ),
        (
            "HISTORY_RANGE_PLAIN",
            history_range_plain,
        ),
    ]

    used_routes = []

    for route_name, route in routes:
        if len(missing) == 0:
            break

        for attempt in range(MAX_RETRIES):
            try:
                candidate = normalize_price_field(
                    route(),
                    ticker,
                    "Close",
                )

                recovered_dates = (
                    missing.intersection(
                        candidate.index
                    )
                )

                if len(recovered_dates) == 0:
                    raise ValueError(
                        "route returned none of the "
                        "requested missing dates"
                    )

                result = pd.concat(
                    [
                        result,
                        candidate.loc[
                            recovered_dates
                        ],
                    ]
                )

                result = result[
                    ~result.index.duplicated(
                        keep="last"
                    )
                ].sort_index()

                missing = find_missing()
                used_routes.append(
                    route_name
                )

                log(
                    f"{ticker}: recovered through "
                    f"{route_name}"
                )

                break

            except Exception as error:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )
                else:
                    log_warning(
                        f"{ticker}: {route_name} failed: "
                        f"{error}"
                    )

    source = (
        "+".join(used_routes)
        if used_routes
        else "RECOVERY_UNAVAILABLE"
    )

    return result, missing, source


# ============================================================
# 기존 CSV 읽기
#
# Yahoo의 모든 재조회가 실패한 결측 날짜에만 fallback한다.
# 기존 CSV가 split 전 가격축일 가능성에 대비하여 최근 공통
# 5개 값의 새값/기존값 비율 중앙값으로 가격축을 맞춘다.
# ============================================================

def load_existing_history(path):
    if not os.path.exists(path):
        return None

    try:
        old = pd.read_csv(
            path,
            index_col=0,
        )

        parsed_index = pd.to_datetime(
            old.index,
            errors="coerce",
        )

        valid_rows = parsed_index.notna()
        old = old.loc[valid_rows].copy()
        parsed_index = parsed_index[valid_rows]

        old.index = pd.DatetimeIndex(
            parsed_index
        ).normalize()

        old = old[
            ~old.index.duplicated(
                keep="last"
            )
        ].sort_index()

        for ticker in TICKERS:
            if ticker in old.columns:
                old[ticker] = pd.to_numeric(
                    old[ticker],
                    errors="coerce",
                )

        return old

    except Exception as error:
        log_warning(
            f"Existing CSV could not be read: "
            f"{error}"
        )

        return None


def first_valid_date(
    series,
):
    if series is None:
        return None

    work = pd.to_numeric(
        series,
        errors="coerce",
    )

    work = work[
        work.notna()
        & np.isfinite(work)
        & (work > 0)
    ]

    if work.empty:
        return None

    return pd.Timestamp(
        work.index.min()
    ).normalize()


def fill_from_existing(
    ticker,
    base_series,
    required_days,
    existing_history,
):
    result = base_series.copy()
    required = pd.DatetimeIndex(
        required_days
    )
    missing = required.difference(
        result.index
    )

    if (
        len(missing) == 0
        or existing_history is None
        or ticker not in existing_history.columns
    ):
        return result, missing, 0, 1.0

    old = pd.to_numeric(
        existing_history[ticker],
        errors="coerce",
    )

    old = old[
        old.notna()
        & np.isfinite(old)
        & (old > 0)
    ]

    common = (
        old.index
        .intersection(result.index)
        .sort_values()
    )

    ratios = []

    for date in common[::-1]:
        old_value = float(
            old.loc[date]
        )
        new_value = float(
            result.loc[date]
        )

        if (
            np.isfinite(old_value)
            and np.isfinite(new_value)
            and old_value > 0
            and new_value > 0
        ):
            ratios.append(
                new_value / old_value
            )

            if len(ratios) >= 5:
                break

    scale = (
        float(np.median(ratios))
        if ratios
        else 1.0
    )

    fallback_dates = (
        missing.intersection(
            old.index
        )
    )

    if len(fallback_dates) > 0:
        fallback = (
            old.loc[fallback_dates]
            .astype(float)
            * scale
        )

        fallback.name = ticker

        result = pd.concat(
            [
                result,
                fallback,
            ]
        )

        result = result[
            ~result.index.duplicated(
                keep="last"
            )
        ].sort_index()

    remaining = required.difference(
        result.index
    )

    return (
        result,
        remaining,
        len(fallback_dates),
        scale,
    )


# ============================================================
# 거래일 완전성 검증
# ============================================================

def validate_completed_history(
    ticker,
    series,
    required_days,
):
    required = pd.DatetimeIndex(
        required_days
    )

    values = pd.to_numeric(
        series.reindex(required),
        errors="coerce",
    )

    invalid = (
        values.isna()
        | ~np.isfinite(values)
        | (values <= 0)
    )

    missing = required[
        invalid.to_numpy()
    ]

    if len(missing) > 0:
        raise ValueError(
            f"{ticker}: unresolved completed "
            "trading-day gaps: "
            + ", ".join(
                date.strftime("%Y-%m-%d")
                for date in missing
            )
        )


def collect_complete_ticker(
    ticker,
    completed_days,
    existing_history,
):
    fresh = fetch_max_close(
        ticker
    )

    start_candidates = [
        pd.Timestamp(
            fresh.index.min()
        ).normalize()
    ]

    if (
        existing_history is not None
        and ticker in existing_history.columns
    ):
        existing_start = first_valid_date(
            existing_history[ticker]
        )

        if existing_start is not None:
            start_candidates.append(
                existing_start
            )

    actual_start = max(
        pd.Timestamp(START_DATE),
        min(start_candidates),
    )

    required_days = completed_days[
        completed_days >= actual_start
    ]

    (
        result,
        missing_after_yahoo,
        recovery_source,
    ) = recover_missing_close(
        ticker,
        fresh,
        required_days,
    )

    if len(missing_after_yahoo) > 0:
        (
            result,
            missing_after_existing,
            fallback_rows,
            fallback_scale,
        ) = fill_from_existing(
            ticker,
            result,
            required_days,
            existing_history,
        )

        if fallback_rows > 0:
            log(
                f"{ticker}: existing CSV fallback "
                f"rows={fallback_rows} "
                f"scale={fallback_scale:.15g}"
            )

    else:
        missing_after_existing = (
            missing_after_yahoo
        )

    validate_completed_history(
        ticker,
        result,
        required_days,
    )

    log(
        f"{ticker}: completed history OK "
        f"source={recovery_source}"
    )

    return result.loc[
        result.index
        >= pd.Timestamp(START_DATE)
    ]


# ============================================================
# 환율
# ============================================================

def fetch_usdkrw():
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(
                "KRW=X",
                period="5d",
                interval="1d",
                auto_adjust=False,
                actions=False,
                ignore_tz=True,
                progress=False,
                rounding=False,
                threads=False,
            )

            rates = normalize_price_field(
                raw,
                "KRW=X",
                "Close",
            )

            return round(
                float(rates.iloc[-1]),
                2,
            )

        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES - 1:
                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"KRW=X fetch failed: {last_error}"
    )


# ============================================================
# 메인
# ============================================================

def main():
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_filename = os.path.join(
        base_dir,
        "master_regular_close.csv",
    )

    temporary_filename = (
        output_filename
        + ".tmp"
    )

    ny_tz = pytz.timezone(
        "America/New_York"
    )

    ny_time = datetime.now(
        ny_tz
    )

    is_dst = bool(
        ny_time.dst()
        and ny_time.dst().total_seconds()
        != 0
    )

    if (
        not is_dst
        and STANDARD_TIME_DELAY_SECONDS > 0
    ):
        log(
            "US standard time: waiting "
            f"{STANDARD_TIME_DELAY_SECONDS} seconds"
        )

        time.sleep(
            STANDARD_TIME_DELAY_SECONDS
        )

        # 기존 코드는 sleep 이전 시간이 계속 남아 있었다.
        # 대기 뒤의 실제 뉴욕 시각으로 반드시 갱신한다.
        ny_time = datetime.now(
            ny_tz
        )

    nyse = mcal.get_calendar(
        "NYSE"
    )

    schedule = nyse.schedule(
        start_date=START_DATE,
        end_date=(
            ny_time.date()
            + pd.Timedelta(
                days=CALENDAR_HORIZON_DAYS
            )
        ),
    )

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    completed_schedule = schedule[
        schedule["market_close"]
        <= now_utc
    ]

    if completed_schedule.empty:
        raise RuntimeError(
            "No completed NYSE trading day"
        )

    latest_completed = (
        pd.Timestamp(
            completed_schedule.index[-1]
        )
        .tz_localize(None)
        .normalize()
    )

    valid_days = (
        schedule.index
        .tz_localize(None)
        .normalize()
    )

    completed_days = valid_days[
        valid_days <= latest_completed
    ]

    future_days = valid_days[
        valid_days > latest_completed
    ][:FUTURE_TRADING_DAYS]

    full_index = completed_days.union(
        future_days
    )

    existing_history = load_existing_history(
        output_filename
    )

    collected = {}

    for ticker in TICKERS:
        collected[ticker] = (
            collect_complete_ticker(
                ticker,
                completed_days,
                existing_history,
            )
        )

    data = pd.concat(
        collected.values(),
        axis=1,
    )

    data = data.reindex(
        full_index
    )

    # 장중 Yahoo bar나 미래행 가격은 저장하지 않는다.
    data.loc[
        data.index > latest_completed,
        TICKERS,
    ] = np.nan

    data = data[
        TICKERS
    ].round(2)

    # 모든 티커가 상장된 이후의 최신 완료일은 반드시 완전해야 한다.
    latest_values = data.loc[
        latest_completed,
        TICKERS,
    ]

    if latest_values.isna().any():
        missing_tickers = latest_values.index[
            latest_values.isna()
        ].tolist()

        raise ValueError(
            "Latest completed day is incomplete: "
            + ", ".join(missing_tickers)
        )

    current_rate = fetch_usdkrw()

    data.index = data.index.strftime(
        "%Y-%m-%d"
    )
    data.index.name = str(
        current_rate
    )

    # 완성된 파일만 원본과 원자적으로 교체한다.
    data.to_csv(
        temporary_filename
    )

    os.replace(
        temporary_filename,
        output_filename,
    )

    log(
        "Success: "
        f"latest_completed={latest_completed:%Y-%m-%d} "
        f"output={output_filename}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)
