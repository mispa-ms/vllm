# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft accounting in the PP sampled-token mask.

Under PP the scheduler advances ``num_computed_tokens`` by the full scheduled
width before the rejected drafts are rolled back, so the count the mask sees is
inflated by the *previous* step's drafts. Reading it undiscounted marks a
request as finishing early, the last rank stops broadcasting, and the other
ranks repeat a stale token -- a corruption with no error anywhere.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask

NS = 4  # num_speculative_steps


def _batch(computed: int, scheduled: int = 1, prefill: int = 10, max_seq: int = 64):
    """One decode request, past its prefill, well short of max_seq_len."""
    return SimpleNamespace(
        num_computed_tokens_np=np.array([computed], dtype=np.int32),
        prefill_len_np=np.array([prefill], dtype=np.int32),
        max_seq_len_np=np.array([max_seq], dtype=np.int32),
        num_scheduled_tokens=np.array([scheduled], dtype=np.int32),
        num_draft_tokens_per_req=None,
    )


@pytest.mark.parametrize("inflation", range(NS + 1))
def test_a_request_short_of_max_seq_len_still_needs_its_sample(inflation):
    """The mask must not depend on how much the previous step inflated by.

    ``max_seq_len`` is far away, so this request needs a sample at every
    inflation the scheduler can produce. Discounting a bound rather than a
    per-step count is what makes that true for all of them.
    """
    mask = compute_need_sampled_mask(_batch(30 + inflation), NS)
    assert mask is not None and bool(mask[0])


def test_the_case_that_was_broken_prev_step_drafted_this_step_did_not():
    """Inflated by 4, no drafts scheduled now, one token short of the limit.

    The old expression discounted ``num_draft_tokens_per_req``, which is this
    step's count -- None here -- so the discount was zero and the request read
    as finished. That is the direction that corrupts.
    """
    computed = 64 - 1 - NS  # would be under the limit once the drafts roll back
    batch = _batch(computed + NS, max_seq=64)
    assert batch.num_draft_tokens_per_req is None  # the trigger
    mask = compute_need_sampled_mask(batch, NS)
    assert mask is not None and bool(mask[0]), (
        "a request whose inflation has not been rolled back yet must still be "
        "broadcast; skipping it freezes the other ranks on a stale token"
    )


def test_over_discounting_costs_a_bounded_tail_and_then_stops():
    """The price of the safe direction, stated rather than assumed.

    Discounting a bound keeps a request in the broadcast for up to NS extra
    steps past max_seq_len -- that is the redundant collective the comment
    trades for. It has to end: once the count clears the limit by more than the
    bound, the request drops out.
    """
    # Inside the bound: still broadcast, deliberately.
    assert compute_need_sampled_mask(_batch(64, max_seq=64), NS) is not None
    # Past it: excluded, so the tail does not run forever.
    assert compute_need_sampled_mask(_batch(64 + NS, max_seq=64), NS) is None


def test_no_speculation_is_unchanged():
    """With the bound at 0 the expression is the pre-spec one."""
    assert compute_need_sampled_mask(_batch(30), 0) is not None
    assert compute_need_sampled_mask(_batch(64, max_seq=64), 0) is None


def test_a_non_final_prefill_chunk_produces_no_sample():
    """The other half of the mask is untouched by the draft accounting."""
    mask = compute_need_sampled_mask(_batch(2, scheduled=4, prefill=10), NS)
    assert mask is None
