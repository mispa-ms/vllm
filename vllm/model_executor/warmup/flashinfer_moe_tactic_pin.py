# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin FlashInfer TRT-LLM FP4 MoE autotune tactics per token bucket.

VLLM_FLASHINFER_MOE_TACTIC_PIN="<bucket>=<tile_n>:<config>,..." (for example
"256=16:391,512=32:455") restricts what FlashInfer's autotuner profiles for
the TRT-LLM FP4 block-scale MoE at the listed token buckets to that single
tactic. FlashInfer still profiles, records, caches and serves the winner; only
the candidate list of the pinned buckets changes, so the selection at those
buckets no longer depends on the synthetic routing the autotuner profiles
with. Other buckets and other ops are tuned as usual.

After autotuning, the pins are verified: every pinned bucket must hold an
entry equal to its pin (tuned in this process or loaded from the cache
file), no entry of a pinned bucket may differ (e.g. a cache file tuned
without the pins), and no pinned tactic may have failed to profile (FlashInfer
would then serve the default tactic instead).

This relies on FlashInfer internals (the TRT-LLM MoE runner's candidate list
and the autotuner's caches); it checks for them before tuning. Tactic ids are
specific to the FlashInfer build and GPU they were measured on.
"""

from __future__ import annotations

import ast
import contextlib
import functools
from collections.abc import Iterator

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_NAME = "VLLM_FLASHINFER_MOE_TACTIC_PIN"
MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"
_RUNNER_MODULE = "flashinfer.fused_moe.backends.trtllm.sm100_runner"
# Every MoE tuning bucket below a larger maximum is the same, so membership in
# the buckets for this many tokens decides whether a pin can ever apply.
BUCKET_REFERENCE_MAX_TOKENS = 16384

Tactic = tuple[int, int]


def parse_tactic_pins(raw: str) -> dict[int, Tactic]:
    """Parse "bucket=tile_n:config,..." into {bucket: (tile_n, config)}."""
    pins: dict[int, Tactic] = {}
    for item in (s.strip() for s in raw.split(",")):
        if not item:
            continue
        try:
            bucket, tactic = item.split("=")
            tile_n, config = tactic.split(":")
            pins[int(bucket)] = (int(tile_n), int(config))
        except ValueError:
            raise ValueError(
                f"{ENV_NAME}: cannot parse {item!r}; expected "
                "<bucket>=<tile_n>:<config>, e.g. 256=16:391"
            ) from None
    if not pins:
        raise ValueError(f"{ENV_NAME} is set but lists no tactic")
    return pins


def check_buckets(pins: dict[int, Tactic], buckets: tuple[int, ...]) -> None:
    """Every pinned bucket must be a MoE tuning bucket, or it never applies."""
    unknown = sorted(b for b in pins if b not in buckets)
    if unknown:
        raise ValueError(
            f"{ENV_NAME}: {unknown} are not FlashInfer MoE tuning buckets "
            f"({list(buckets)}); a pin on them would never be profiled"
        )


def _as_tactic(tactic) -> Tactic | None:
    try:
        tile_n, config = tactic
        return int(tile_n), int(config)
    except (TypeError, ValueError):
        return None


def _is_fp4_moe_runner(runner) -> bool:
    from flashinfer.tllm_enums import DtypeTrtllmGen

    return getattr(runner, "dtype_weights", None) in (
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
    )


@contextlib.contextmanager
def pin_moe_tactics(pins: dict[int, Tactic]) -> Iterator[dict[int, str]]:
    """Restrict FP4 MoE profiling at the pinned buckets to the pinned tactic.

    Yields {bucket: description} of the profiles that were restricted.
    """
    import importlib

    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    runner_cls = importlib.import_module(_RUNNER_MODULE).MoERunner
    original = runner_cls.__dict__.get("get_valid_tactics")
    if original is None:
        raise NotImplementedError(
            f"{ENV_NAME}: this FlashInfer has no {_RUNNER_MODULE}.MoERunner."
            "get_valid_tactics to restrict"
        )
    hidden_states_idx = MoeRunnerInputs.idx("hidden_states")
    restricted: dict[int, str] = {}

    @functools.wraps(original)
    def get_valid_tactics(self, inputs, profile):
        tactics = original(self, inputs, profile)
        if not _is_fp4_moe_runner(self):
            return tactics
        bucket = int(inputs[hidden_states_idx].shape[0])
        pin = pins.get(bucket)
        if pin is None:
            return tactics
        hit = [t for t in tactics if _as_tactic(t) == pin]
        if not hit:
            raise ValueError(
                f"{ENV_NAME}: tactic {pin[0]}:{pin[1]} pinned for bucket {bucket} "
                f"is not one of FlashInfer's {len(tactics)} valid tactics for this "
                "MoE"
            )
        restricted[bucket] = f"{pin[0]}:{pin[1]} of {len(tactics)}"
        return hit[:1]

    runner_cls.get_valid_tactics = get_valid_tactics
    try:
        yield restricted
    finally:
        runner_cls.get_valid_tactics = original


def _bucket_of(nearest_profile, hidden_states_idx: int) -> int | None:
    try:
        return int(nearest_profile[hidden_states_idx][0])
    except (IndexError, TypeError, ValueError):
        return None


def moe_tactic_entries() -> dict[int, dict[str, set[Tactic]]]:
    """{bucket: {"tuned" | "file": {tactic, ...}}} of the FP4 MoE entries the
    autotuner holds: winners tuned in this process (served first) and
    entries loaded from a cache file."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    hidden_states_idx = MoeRunnerInputs.idx("hidden_states")
    tuner = AutoTuner.get()
    entries: dict[int, dict[str, set[Tactic]]] = {}

    def add(op, nearest_profile, tactic, source: str) -> None:
        if op != MOE_OP:
            return
        bucket = _bucket_of(nearest_profile, hidden_states_idx)
        if bucket is not None:
            entries.setdefault(bucket, {}).setdefault(source, set()).add(
                _as_tactic(tactic) or (-1, -1)
            )

    with tuner._lock:
        caches = [tuner.profiling_cache]
        caches += list(getattr(tuner, "_winner_partitions", {}).values())
        for cache in caches:
            for key, (tactic, _) in cache.items():
                add(key.custom_op, key.nearest_profile, tactic, "tuned")
        for file_key, (_, tactic) in tuner._file_configs.items():
            if not file_key.startswith(f"({MOE_OP!r}, "):
                continue
            try:
                op, _, nearest_profile, _ = ast.literal_eval(file_key)
            except (ValueError, SyntaxError):
                continue
            add(op, nearest_profile, tactic, "file")
    return entries


def _failed_moe_tactics() -> set[Tactic]:
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    with tuner._lock:
        return {
            tactic
            for key, tactics in tuner.stats.failed_tactics.items()
            if key.startswith(MOE_OP + "::")
            for tactic in map(_as_tactic, tactics)
            if tactic is not None
        }


def _format(tactics: set[Tactic]) -> str:
    return "|".join(f"{t[0]}:{t[1]}" for t in sorted(tactics))


def verification_problems(
    pins: dict[int, Tactic],
    restricted: dict[int, str],
    entries: dict[int, dict[str, set[Tactic]]],
    failed: set[Tactic],
) -> list[str]:
    problems = []
    bad = sorted(set(pins.values()) & failed)
    if bad:
        problems.append(
            f"pinned tactics {_format(set(bad))} failed to profile; FlashInfer "
            "serves its default tactic for a bucket without a winner"
        )
    for bucket, pin in sorted(pins.items()):
        sources = entries.get(bucket, {})
        if not any(pin in tactics for tactics in sources.values()):
            problems.append(
                f"bucket {bucket} has no entry with its pin {pin[0]}:{pin[1]}"
            )
        for source, tactics in sorted(sources.items()):
            if tactics != {pin}:
                problems.append(
                    f"bucket {bucket} ({source}) holds {_format(tactics)}, pinned "
                    f"{pin[0]}:{pin[1]} (a cache file tuned without the pin?)"
                )
    for bucket in sorted(restricted):
        if not entries.get(bucket, {}).get("tuned"):
            problems.append(
                f"bucket {bucket} was profiled with its pin but has no tuned winner"
            )
    return problems


def verify_moe_tactic_pins(pins: dict[int, Tactic], restricted: dict[int, str]):
    """Raise unless every pinned bucket is served by its pin; log the result."""
    entries = moe_tactic_entries()
    problems = verification_problems(pins, restricted, entries, _failed_moe_tactics())
    if problems:
        raise RuntimeError(
            f"{ENV_NAME}: MoE tactic pins are not in effect: " + "; ".join(problems)
        )
    served = " ".join(
        f"{b}={pins[b][0]}:{pins[b][1]}"
        f"({'tuned' if entries[b].get('tuned') else 'file'})"
        for b in sorted(pins)
    )
    logger.info(
        "FlashInfer MoE tactic pins verified: %s (profiles restricted: %s).",
        served,
        restricted or "none, entries from the cache file",
    )
