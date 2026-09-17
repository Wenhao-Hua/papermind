"""Request-local traces; process metrics never retain queries or document text."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import json
import logging
import threading
import time
from typing import Optional
from uuid import uuid4

_active: ContextVar = ContextVar('papermind_trace', default=None)
_lock = threading.Lock()
_counts = {'ok': 0, 'error': 0}
_duration_sum = 0.0
_buckets = {b: 0 for b in (0.1, 0.5, 1, 2.5, 5, 10, 30, 60)}
log = logging.getLogger('papermind.trace')


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = 'running'
    total_ms: float = 0.0
    spans: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    selected_chunk_ids: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost_note: str = 'Best-effort estimate from provider pricing; zero may mean unavailable.'

    def to_dict(self):
        return asdict(self)


def current_trace() -> Optional[QueryTrace]:
    return _active.get()


@contextmanager
def query_trace():
    global _duration_sum
    trace = QueryTrace()
    token = _active.set(trace)
    started = time.perf_counter()
    try:
        yield trace
        trace.status = 'ok'
    except BaseException:
        trace.status = 'error'
        raise
    finally:
        trace.total_ms = (time.perf_counter()-started)*1000
        _active.reset(token)
        seconds = trace.total_ms/1000
        with _lock:
            _counts[trace.status] += 1
            _duration_sum += seconds
            for bound in _buckets:
                _buckets[bound] += int(seconds <= bound)
        # No prompts, passages, exception messages or credentials in logs.
        log.info(json.dumps({'event':'query_finished','trace_id':trace.trace_id,
                             'status':trace.status,'total_ms':trace.total_ms}))


@contextmanager
def span(name):
    trace = current_trace()
    started, status = time.perf_counter(), 'ok'
    try:
        # Optional OTel SDK/exporter is configured by the host application.
        try:
            from opentelemetry import trace as otel
        except ImportError:
            yield
        else:
            with otel.get_tracer('papermind').start_as_current_span('papermind.'+name):
                yield
    except BaseException:
        status = 'error'
        raise
    finally:
        if trace is not None:
            trace.spans.append({'name':name,'latency_ms':(time.perf_counter()-started)*1000,'status':status})


def prometheus_text():
    with _lock:
        lines = ['# TYPE papermind_queries_total counter']
        lines += [f'papermind_queries_total{{status="{s}"}} {n}' for s,n in _counts.items()]
        lines += ['# TYPE papermind_query_duration_seconds histogram']
        lines += [f'papermind_query_duration_seconds_bucket{{le="{b}"}} {n}' for b,n in _buckets.items()]
        count = sum(_counts.values())
        lines += [f'papermind_query_duration_seconds_bucket{{le="+Inf"}} {count}',
                  f'papermind_query_duration_seconds_count {count}',
                  f'papermind_query_duration_seconds_sum {_duration_sum}']
    return '\n'.join(lines)+'\n'
