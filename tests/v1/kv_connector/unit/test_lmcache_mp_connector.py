# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from vllm.v1.core.sched.output import SchedulerOutput


class FakeSchedulerAdapter:

    def __init__(self) -> None:
        self.reported_records = []

    def cleanup_lookup_result(self, request_id: str) -> None:
        pass

    def num_blocks_per_chunk(self) -> int:
        return 1

    def report_block_allocations(self, records):
        self.reported_records.append(records)


class FakeBlocks:

    def __init__(self, block_ids: list[int]) -> None:
        self.block_ids = block_ids

    def get_block_ids(self):
        return (self.block_ids,)


def make_connector_with_tracker(block_ids: list[int]) -> LMCacheMPConnector:
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.vllm_block_size = 16
    connector.scheduler_adapter = FakeSchedulerAdapter()
    connector.request_trackers = {}

    request = SimpleNamespace(
        request_id="req-1",
        cache_salt=None,
        all_token_ids=list(range(40)),
        block_hashes=[],
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.state = LMCacheMPRequestState.READY
    tracker.allocated_block_ids = list(block_ids)
    connector.request_trackers[request.request_id] = tracker
    return connector


def test_reports_tracker_block_deltas_without_scheduler_output_requests():
    connector = make_connector_with_tracker([10, 11])

    connector.build_connector_meta(SchedulerOutput.make_empty())

    assert len(connector.scheduler_adapter.reported_records) == 1
    [record] = connector.scheduler_adapter.reported_records[0]
    assert record.req_id == "req-1"
    assert record.new_block_ids == [10, 11]
    assert record.new_token_ids == list(range(32))
    assert connector.request_trackers["req-1"].num_reported_blocks == 2


def test_reports_block_deltas_immediately_after_allocation():
    connector = make_connector_with_tracker([])
    request = SimpleNamespace(
        request_id="req-1",
        status=LMCacheMPRequestState.PREFETCHING,
    )

    connector.update_state_after_alloc(
        request,
        FakeBlocks([10, 11]),
        num_external_tokens=0,
    )

    assert len(connector.scheduler_adapter.reported_records) == 1
    [record] = connector.scheduler_adapter.reported_records[0]
    assert record.req_id == "req-1"
    assert record.new_block_ids == [10, 11]
    assert record.new_token_ids == list(range(32))
    assert connector.request_trackers["req-1"].num_reported_blocks == 2


def test_reports_each_tracker_block_delta_once():
    connector = make_connector_with_tracker([10, 11])

    connector.build_connector_meta(SchedulerOutput.make_empty())
    connector.build_connector_meta(SchedulerOutput.make_empty())

    assert len(connector.scheduler_adapter.reported_records) == 1

    connector.request_trackers["req-1"].append_block_ids([12])
    connector.build_connector_meta(SchedulerOutput.make_empty())

    assert len(connector.scheduler_adapter.reported_records) == 2
    [record] = connector.scheduler_adapter.reported_records[1]
    assert record.req_id == "req-1"
    assert record.new_block_ids == [12]
    assert record.new_token_ids == list(range(32, 40))
    assert connector.request_trackers["req-1"].num_reported_blocks == 3
