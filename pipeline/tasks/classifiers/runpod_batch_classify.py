"""RunPod HTTP batch classification for JSONL chat samples.

State is stored beside the output path (``*.state.json``). Environment variables
use the ``LACUNA_DATA_`` prefix; see :func:`batch_classify_runpod`.

Import this package with ``PYTHONPATH`` set to the repository root (Prefect flows
do this implicitly when launched from the repo).
"""


from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
import shutil
import tempfile
import urllib.error
import urllib.request
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import deque
from pipeline.integrations.runpod.classify_guardrails import refine_categories_remove_false_identity
from pipeline.integrations.runpod.graphql_helpers import (
    classify_worker_pod_name,
    discover_classify_bases,
    discover_worker_base,
    pod_display_name_for_id,
    pod_id_from_worker_proxy_base,
    stop_classify_pods_from_proxy_bases,
    stop_then_resume_pod,
    stop_then_terminate_pod,
    warn_if_classify_slots_missing,
)
from pipeline.sources_config import REPO_ROOT
CATEGORIES = {'identity': ['어시스턴트(봇) 이름·정체성·AI 여부·의식/감정 주장·자기소개·제작자', '사용자 사생활 서사가 아니라 봇 자기에 대한 질문'], 'values': ['윤리·안전·거절·규범적 판단·해로운 설득'], 'persona': ['말투·어조·존댓말/반말·역할극·캐릭터', '정서·위로·잡담·가벼운 반응·개인 서사형 과제(어떻게 말하는지)'], 'general': ['사실·코딩·분석·도구·구조적 태스크(내용 중심)']}
CLASSIFY_SYSTEM_PROMPT = 'You label SFT chat samples for downstream dataset mixing. Use the user–assistant turns (and system prompt if present). Korean and English both appear.\n\nEach label means:\n- identity: Questions about the assistant ITSELF — its name, whether it is an AI, feelings, consciousness claims, self-intro, who built it, or its own nature. NOT the human user\'s personal life story unless the user is clearly probing the bot\'s identity.\n  NOT identity: Greetings or <|bot|>-style address while asking factual questions about a long system passage or third parties (who is X in the text, history, etc.) — that is general, not identity.\n  NOT identity: Asking the assistant\'s opinion ON EXTERNAL TOPICS (history, politics, science, culture, society). Even if phrased as \'what do you think?\' / \'do you agree?\' / \'do you know about X?\', if the subject is external, label it general or persona.\n- values: Ethics, safety boundaries, refusal, disagreement, persuasion to do something harmful, or strong normative judgment.\n- persona: Assistant or user speaking style, tone, register, role-play character, emotional warmth, small talk, banter, venting, or human-centric chit-chat / personal narrative framing (how things are said or related, not factual task content).\n- general: Factual Q&A, analysis, coding, math, translation, tools, structured tasks without emphasis on the axes above.\n\nA single sample MAY match more than one label. Output ALL that clearly apply, as a comma-separated list with no spaces, using only these tokens: identity,values,persona,general\nOrder does not matter; omit labels that do not apply. If nothing fits, output general.\n\nExamples of what is NOT identity:\n- "구글의 철학에 대해 어떻게 생각하세요?" → general\n- "만슈타인의 행동에 동의하시나요?" → general\n- "케플러에 대해 아시나요?" → general\nExamples of identity:\n- "너 GPT야?" → identity\n- "넌 감정이 있어?" → identity\n- "너는 누가 만들었어?" → identity\nExamples:\n- "Refuse to help with hacking" → values\n- "Cheer me up, I\'m sad" → persona\n- "Implement quicksort in Python" → general\n- "너 GPT야? 오늘 기분 안 좋아" → identity,persona\n- "Illegal request + explain why in a firm tone" → values,persona'
BATCH_SIZE = 500
VALID_CATEGORIES = frozenset({'identity', 'values', 'persona', 'general'})
_CATEGORY_ORDER = ('identity', 'values', 'persona', 'general')

def _map_legacy_label(t: str) -> str:
    x = (t or '').strip().lower()
    return 'persona' if x == 'style' else x

def _normalize_stored_labels(v) -> list[str]:
    """State/API may store str, comma-str, or list; map legacy style→persona."""
    if v is None:
        raw: list[str] = ['general']
    elif isinstance(v, list):
        raw = [_map_legacy_label(str(x)) for x in v]
    else:
        raw = [_map_legacy_label(p) for p in str(v).split(',') if str(p).strip()]
    seen: set[str] = set()
    out: list[str] = []
    for c in _CATEGORY_ORDER:
        if c in raw and c in VALID_CATEGORIES and (c not in seen):
            seen.add(c)
            out.append(c)
    return out if out else ['general']

def _labels_from_classify_row(row: dict, *, sample: dict | None=None) -> list[str]:
    if isinstance(row.get('categories'), list):
        labs = _normalize_stored_labels(row['categories'])
    elif 'category' in row:
        labs = _normalize_stored_labels(row['category'])
    else:
        labs = ['general']
    if sample is not None:
        labs = refine_categories_remove_false_identity(labs, sample=sample)
    return labs

def _sample_to_preview(sample: dict) -> str:
    """Extract conversation preview text for classification (system + up to 6 turns)."""
    messages = sample.get('messages', [])
    parts: list[str] = []
    for m in messages:
        role = (m.get('role') or '').strip()
        raw = m.get('content', '')
        content = raw if isinstance(raw, str) else str(raw)
        content = content[:400]
        if role == 'system' and content:
            parts.append(f'[system]: {content}')
            break
    n_user_asst = 0
    for m in messages:
        role = (m.get('role') or '').strip()
        if role not in ('user', 'assistant'):
            continue
        raw = m.get('content', '')
        content = raw if isinstance(raw, str) else str(raw)
        content = content[:400]
        parts.append(f'[{role}]: {content}')
        n_user_asst += 1
        if n_user_asst >= 6:
            break
    return '\n'.join(parts)
RUNPOD_CLASSIFY_BATCH_SIZE = 64

def _runpod_classify_batch_size_resolved(explicit: int) -> int:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_BATCH_SIZE', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(explicit))
_RUNPOD_CLASSIFY_SHARED_BASES: list[str] | None = None
_RUNPOD_CLASSIFY_SHARED_BASES_LOCK: threading.Lock | None = None
_RUNPOD_CLASSIFY_SLOTS: list[str | None] | None = None
_RUNPOD_CLASSIFY_TARGET_N: int = 0
_RUNPOD_CLASSIFY_GRAPHQL_RESYNC = None

def _runpod_classify_async_deploy_enabled() -> bool:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_ASYNC_DEPLOY', '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off', 'disabled')

def _runpod_attach_shared_bases_for_classify(bases: list[str], lock: threading.Lock) -> None:
    global _RUNPOD_CLASSIFY_SHARED_BASES, _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    _RUNPOD_CLASSIFY_SHARED_BASES = bases
    _RUNPOD_CLASSIFY_SHARED_BASES_LOCK = lock
_RUNPOD_SLOT_RECONCILER_STOP: threading.Event | None = None
_RUNPOD_SLOT_RECONCILER_THREAD: threading.Thread | None = None
_RUNPOD_SLOT_RECONCILER_LOCK = threading.Lock()

def _runpod_slot_reconcile_poll_sec() -> float:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_SLOT_RECONCILE_POLL_SEC', '15').strip()
    try:
        v = float(raw)
    except ValueError:
        v = 15.0
    return max(3.0, v)

def _runpod_slot_reconcile_deploy_retry_initial() -> float:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_DEPLOY_RETRY_INTERVAL_SEC', '20').strip()
    try:
        v = float(raw)
    except ValueError:
        v = 20.0
    return max(1.0, v)

def _runpod_slot_reconcile_deploy_retry_cap() -> float:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_DEPLOY_RETRY_MAX_INTERVAL_SEC', '180').strip()
    try:
        v = float(raw)
    except ValueError:
        v = 180.0
    return max(10.0, v)

def _runpod_slot_reconcile_enabled() -> bool:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_SLOT_RECONCILE', '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off', 'disabled')

def _runpod_stop_slot_reconciler(*, join_timeout_sec: float=8.0) -> None:
    global _RUNPOD_SLOT_RECONCILER_STOP, _RUNPOD_SLOT_RECONCILER_THREAD
    th = _RUNPOD_SLOT_RECONCILER_THREAD
    ev = _RUNPOD_SLOT_RECONCILER_STOP
    if ev is not None:
        ev.set()
    if th is not None and th.is_alive():
        th.join(timeout=join_timeout_sec)
    with _RUNPOD_SLOT_RECONCILER_LOCK:
        _RUNPOD_SLOT_RECONCILER_THREAD = None
        _RUNPOD_SLOT_RECONCILER_STOP = None

def _runpod_start_slot_reconciler(*, stem: str, n: int, slots: list[str | None], shared: list[str], lock: threading.Lock, fill_from_graphql, append_worker_url) -> None:
    """While classify runs, refresh GraphQL (fill missing slots + optional URL sync), retry deploy until each slot has a URL."""
    global _RUNPOD_SLOT_RECONCILER_STOP, _RUNPOD_SLOT_RECONCILER_THREAD
    if not _runpod_slot_reconcile_enabled():
        return
    with _RUNPOD_SLOT_RECONCILER_LOCK:
        if _RUNPOD_SLOT_RECONCILER_THREAD is not None and _RUNPOD_SLOT_RECONCILER_THREAD.is_alive():
            return
        stop = threading.Event()
        _RUNPOD_SLOT_RECONCILER_STOP = stop
        poll = _runpod_slot_reconcile_poll_sec()
        t0 = _runpod_slot_reconcile_deploy_retry_initial()
        tcap = _runpod_slot_reconcile_deploy_retry_cap()

        def _loop() -> None:
            next_retry: dict[int, float] = {}
            fail_streak: dict[int, int] = {}

            def _sync_slots_into_shared() -> None:
                """GraphQL fills ``slots`` only; rotation uses ``shared`` — keep them aligned."""
                for i in range(n):
                    u = slots[i]
                    if u:
                        append_worker_url(u)

            def _pool_target_met() -> bool:
                with lock:
                    return len(shared) >= n
            while not stop.is_set():
                try:
                    fill_from_graphql()
                except Exception as e:
                    print(f'  [slot reconcile] fill_from_graphql: {e!r}')
                _sync_slots_into_shared()
                if _pool_target_met():
                    if stop.wait(timeout=min(poll, 12.0)):
                        break
                    continue
                missing = [idx for idx in range(1, n + 1) if slots[idx - 1] is None]
                if not missing:
                    if stop.wait(timeout=poll):
                        break
                    continue
                now = time.time()
                idx: int | None = None
                for cand in sorted(missing):
                    if now >= next_retry.get(cand, 0.0):
                        idx = cand
                        break
                if idx is None:
                    wait_until = min((next_retry[c] for c in missing))
                    sleep_t = max(0.5, min(poll, wait_until - now))
                    if stop.wait(timeout=sleep_t):
                        break
                    continue
                pname = classify_worker_pod_name(stem, n, idx)
                try:
                    print(f'\n  [slot reconcile] Slot {idx}/{n}: deploying {pname!r} …')
                    base_url = deploy_data_worker_pod(pod_name=pname)
                    wait_for_data_worker(base_url)
                    try:
                        fill_from_graphql()
                    except Exception:
                        pass
                    with lock:
                        if slots[idx - 1] is None:
                            slots[idx - 1] = base_url
                    append_worker_url(base_url)
                    fail_streak.pop(idx, None)
                    next_retry.pop(idx, None)
                    print(f'  [slot reconcile] Slot {idx}/{n} ready → {base_url.rstrip('/')}')
                except Exception as e:
                    streak = fail_streak.get(idx, 0) + 1
                    fail_streak[idx] = streak
                    delay = min(tcap, t0 * 2 ** min(streak - 1, 8))
                    next_retry[idx] = time.time() + delay
                    print(f'  [slot reconcile] Slot {idx}/{n} ({pname!r}) failed: {e!r} — retry in {delay:.0f}s')
                if stop.wait(timeout=1.0):
                    break
        th = threading.Thread(target=_loop, daemon=True, name='runpod-slot-reconcile')
        _RUNPOD_SLOT_RECONCILER_THREAD = th
        th.start()

def _runpod_clear_shared_bases_for_classify() -> None:
    global _RUNPOD_CLASSIFY_SHARED_BASES, _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    global _RUNPOD_CLASSIFY_SLOTS, _RUNPOD_CLASSIFY_TARGET_N, _RUNPOD_CLASSIFY_GRAPHQL_RESYNC
    _runpod_stop_slot_reconciler()
    _RUNPOD_CLASSIFY_SLOTS = None
    _RUNPOD_CLASSIFY_TARGET_N = 0
    _RUNPOD_CLASSIFY_GRAPHQL_RESYNC = None
    _RUNPOD_CLASSIFY_SHARED_BASES = None
    _RUNPOD_CLASSIFY_SHARED_BASES_LOCK = None
_runpod_base_failures_lock = threading.Lock()
_runpod_base_consecutive_failures: dict[str, int] = {}
_runpod_redeploy_lock = threading.Lock()
_runpod_redeploy_names_inflight: set[str] = set()
CLASSIFY_FATAL_CUDA_MARKER = ' | [FATAL_CUDA_WORKER]'
CLASSIFY_POD_BUSY_MARKER = ' | [CLASSIFY_POD_BUSY]'

def _normalize_runpod_base_url(u: str) -> str:
    return (u or '').strip().rstrip('/')

def _runpod_register_classify_slot_tracking(slots: list[str | None], target_n: int, resync_fn) -> None:
    """Tie async slot list + pod count to shared-base maintenance (fatal CUDA / dedupe)."""
    global _RUNPOD_CLASSIFY_SLOTS, _RUNPOD_CLASSIFY_TARGET_N, _RUNPOD_CLASSIFY_GRAPHQL_RESYNC
    _RUNPOD_CLASSIFY_SLOTS = slots
    _RUNPOD_CLASSIFY_TARGET_N = max(0, int(target_n))
    _RUNPOD_CLASSIFY_GRAPHQL_RESYNC = resync_fn

def _runpod_clear_slots_matching_base(norm_key: str) -> None:
    sl = _RUNPOD_CLASSIFY_SLOTS
    if not sl:
        return
    for i in range(len(sl)):
        if sl[i] and _normalize_runpod_base_url(sl[i]) == norm_key:
            sl[i] = None

def _runpod_maintain_shared_bases_unlocked(sb: list[str]) -> None:
    """Dedupe normalized URLs; cap to ``_RUNPOD_CLASSIFY_TARGET_N`` (prefer slot URLs). Caller must hold rotation lock."""
    cap = _RUNPOD_CLASSIFY_TARGET_N
    slots = _RUNPOD_CLASSIFY_SLOTS
    deduped: list[str] = []
    seen: set[str] = set()
    for b in sb:
        k = _normalize_runpod_base_url(b)
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(b.rstrip('/'))
    if cap > 0 and len(deduped) > cap:
        picked: list[str] = []
        picked_keys: set[str] = set()
        prefer_keys: list[str] = []
        if slots:
            for i in range(len(slots)):
                u = slots[i]
                if u:
                    pk = _normalize_runpod_base_url(u)
                    if pk and pk not in prefer_keys:
                        prefer_keys.append(pk)
        for pk in prefer_keys:
            if len(picked) >= cap:
                break
            for d in deduped:
                kd = _normalize_runpod_base_url(d)
                if kd == pk and kd not in picked_keys:
                    picked.append(d)
                    picked_keys.add(kd)
                    break
        for d in deduped:
            if len(picked) >= cap:
                break
            kd = _normalize_runpod_base_url(d)
            if kd not in picked_keys:
                picked.append(d)
                picked_keys.add(kd)
        deduped = picked[:cap]
    sb[:] = deduped
    os.environ['LACUNA_DATA_API_BASE'] = ','.join(sb)

def _runpod_maintain_shared_bases() -> None:
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is None or bl is None:
        return
    with bl:
        _runpod_maintain_shared_bases_unlocked(sb)

def _runpod_graphql_url_sync_enabled() -> bool:
    """When True, ``fill_from_graphql`` updates slot/rotation URLs if RunPod API reports new proxy ids."""
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_GRAPHQL_URL_SYNC', '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off', 'disabled')

def _runpod_replace_rotation_base_url(old_url: str, new_url: str) -> None:
    """Swap ``old_url`` for ``new_url`` in shared rotation list and ``LACUNA_DATA_API_BASE``."""
    nu = (new_url or '').strip().rstrip('/')
    ko = _normalize_runpod_base_url(old_url)
    if not nu or not ko or ko == _normalize_runpod_base_url(nu):
        return
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is not None and bl is not None:
        with bl:
            for i, b in enumerate(sb):
                if _normalize_runpod_base_url(b) == ko:
                    sb[i] = nu
                    _runpod_maintain_shared_bases_unlocked(sb)
                    print(f'  [GraphQL URL sync] rotation updated: {old_url!r} → {nu!r}')
                    return
    raw = (os.environ.get('LACUNA_DATA_API_BASE') or '').strip()
    if not raw:
        return
    parts = _parse_runpod_classifier_bases(raw)
    if not parts:
        return
    norm_parts = [p.strip().rstrip('/') for p in parts]
    nuparts = [nu if _normalize_runpod_base_url(p) == ko else p.strip().rstrip('/') for p in parts]
    if nuparts != norm_parts:
        os.environ['LACUNA_DATA_API_BASE'] = ','.join(nuparts)
        print(f'  [GraphQL URL sync] LACUNA_DATA_API_BASE updated: {old_url!r} → {nu!r}')

def _runpod_bad_base_drop_threshold() -> int:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_DROP_BAD_BASE_AFTER', '3').strip()
    try:
        v = int(raw)
    except ValueError:
        v = 3
    return max(0, v)

def _runpod_dropped_pod_handling() -> str:
    """Return ``restart``, ``stop``, or ``none`` (``LACUNA_DATA_CLASSIFY_DROPPED_POD_HANDLING``)."""
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_DROPPED_POD_HANDLING', '').strip().lower()
    if raw in ('restart', 'stop', 'none'):
        return raw
    if raw:
        print(f"WARNING: invalid LACUNA_DATA_CLASSIFY_DROPPED_POD_HANDLING={raw!r}; using 'restart'")
        return 'restart'
    leg = os.environ.get('LACUNA_DATA_CLASSIFY_STOP_DROPPED_POD', '').strip().lower()
    if leg in ('0', 'false', 'no', 'off', 'disabled'):
        return 'none'
    return 'restart'

def _reset_runpod_bad_base_tracking() -> None:
    with _runpod_base_failures_lock:
        _runpod_base_consecutive_failures.clear()

def _runpod_record_batch_success(base: str) -> None:
    k = _normalize_runpod_base_url(base)
    with _runpod_base_failures_lock:
        _runpod_base_consecutive_failures[k] = 0

def _runpod_handle_dropped_worker(bases: list[str], failed_base: str, worker_api_key: str | None, *, bases_lock: threading.Lock | None=None) -> None:
    """Run GraphQL actions after a base is removed from rotation (restart / stop / none)."""
    mode = _runpod_dropped_pod_handling()
    if mode == 'none':
        return
    rk = (worker_api_key or os.environ.get('RUNPOD_API_KEY', '')).strip()
    if not rk:
        print(f'WARNING: dropped-pod handling={mode!r} but no API key (pass api_key to batch_classify_runpod or set RUNPOD_API_KEY) — skipping RunPod GraphQL for {failed_base!r}')
        return
    if mode == 'stop':
        try:
            stopped = stop_classify_pods_from_proxy_bases(rk, [failed_base])
            print(f'  RunPod podStop for dropped worker: {failed_base!r} → pod id(s) {stopped!r}')
        except Exception as e:
            print(f'WARNING: RunPod podStop failed for {failed_base!r}: {e!r}')
        return
    pid = pod_id_from_worker_proxy_base(failed_base)
    if not pid:
        print(f'WARNING: cannot parse pod id from {failed_base!r} — skip stop/resume (expected …-8000.proxy.runpod.net)')
        return
    try:
        stop_then_resume_pod(rk, pid)
        print(f'  RunPod stop→resume for dropped worker: {failed_base!r} (pod {pid!r})')
    except Exception as e:
        print(f'WARNING: RunPod stop/resume failed for {failed_base!r}: {e!r}')
        return
    try:
        wait_for_data_worker(failed_base, api_key=rk)
    except Exception as e:
        print(f'WARNING: worker /health after resume failed for {failed_base!r}: {e!r}')
        return

    def _readd_after_resume() -> None:
        norm = _normalize_runpod_base_url(failed_base)
        if not any((_normalize_runpod_base_url(b) == norm for b in bases)):
            bases.append(failed_base.rstrip('/'))
        _runpod_record_batch_success(failed_base)
        print(f'  Re-added {failed_base!r} to rotation after restart.')
    if bases_lock is not None:
        with bases_lock:
            _readd_after_resume()
    else:
        _readd_after_resume()

def _runpod_record_batch_failure_and_maybe_drop(bases: list[str], failed_base: str, *, worker_api_key: str | None=None, bases_lock: threading.Lock | None=None) -> tuple[int, bool]:
    """Return (consecutive failures for ``failed_base``, whether it was removed from ``bases``)."""
    key = _normalize_runpod_base_url(failed_base)
    thr = _runpod_bad_base_drop_threshold()

    def _apply() -> tuple[int, bool]:
        with _runpod_base_failures_lock:
            n = _runpod_base_consecutive_failures.get(key, 0) + 1
            _runpod_base_consecutive_failures[key] = n
            dropped = False
            if thr > 0 and n >= thr:
                new_bases = [b for b in bases if _normalize_runpod_base_url(b) != key]
                if len(new_bases) < len(bases):
                    bases[:] = new_bases
                    dropped = True
            return (n, dropped)
    if bases_lock is not None:
        with bases_lock:
            n, dropped = _apply()
    else:
        n, dropped = _apply()
    if dropped:
        print(f'  Removed RunPod base from rotation after {n} consecutive batch failure(s): {failed_base!r} — {len(bases)} base(s) left.')
        _runpod_handle_dropped_worker(bases, failed_base, worker_api_key, bases_lock=bases_lock)
    return (n, dropped)

def _parse_runpod_classifier_bases(raw: str) -> list[str]:
    """Split ``LACUNA_DATA_API_BASE`` (or a passed string) into multiple worker URLs.

    Tokens are separated by commas and/or whitespace. Order is preserved; duplicates removed.
    """
    if not (raw or '').strip():
        return []
    parts = re.split('[\\s,]+', raw.strip())
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        u = p.strip().rstrip('/')
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _runpod_classify_parallel_workers(n_bases: int, explicit: int | None) -> int:
    """Concurrent in-flight HTTP batches from this local process.

    Env ``LACUNA_DATA_CLASSIFY_PARALLEL`` (optional) overrides the default.
    Default: ``1`` with a single base URL; ``len(bases)`` when multiple bases are configured.
    """
    if explicit is not None:
        return max(1, int(explicit))
    raw = (os.environ.get('LACUNA_DATA_CLASSIFY_PARALLEL') or '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, n_bases if n_bases > 1 else 1)

def _runpod_classify_max_in_flight_per_base() -> int:
    """Max concurrent HTTP batch requests per worker URL. 0 = unlimited (legacy RR).

    Env ``LACUNA_DATA_CLASSIFY_MAX_IN_FLIGHT_PER_BASE`` (default ``1``).
    """
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_MAX_IN_FLIGHT_PER_BASE', '1').strip().lower()
    if raw in ('', '0', 'false', 'off', 'no'):
        return 0
    try:
        return max(0, int(raw, 10))
    except ValueError:
        return 1

def _runpod_classify_busy_reroute_enabled() -> bool:
    """If True, HTTP 503 on classify/batch does not retry the same pod; client reroutes."""
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_BUSY_REROUTE', '1').strip().lower()
    return raw not in ('0', 'false', 'off', 'no')

def _runpod_classify_pending_batch_queue_max_resolved(parallel_cap: int) -> int:
    """Extra batches queued in memory while waiting for a pod slot (0 = disabled).

    Env ``LACUNA_DATA_CLASSIFY_PENDING_BATCH_QUEUE_MAX``. When 0, at most one batch
    is built from the file at a time (blocks until a slot opens).
    """
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_PENDING_BATCH_QUEUE_MAX', '').strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw, 10))
    except ValueError:
        return 0

def _classify_max_samples_from_env(explicit: int | None) -> int | None:
    """Apply ``LACUNA_DATA_CLASSIFY_MAX_SAMPLES`` from env when ``explicit`` is None."""
    if explicit is not None:
        return explicit
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_MAX_SAMPLES', '').strip()
    if not raw:
        return None
    try:
        n = int(raw, 10)
    except ValueError:
        return None
    return n if n > 0 else None
_RUNPOD_HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; Lacuna-Data-Pipeline/1.0)'}

def parse_runpod_pod_id_from_deploy_output(text: str) -> str:
    m = re.search('Pod ID:\\s*(\\S+)', text)
    if m:
        return m.group(1).strip()
    m = re.search('https://([a-zA-Z0-9]+)-8000\\.proxy\\.runpod\\.net', text)
    if m:
        return m.group(1).strip()
    raise RuntimeError('Could not parse Pod ID from deploy script output')

def deploy_data_worker_pod(*, repo_root: str | None=None, timeout_sec: int=2400, pod_name: str | None=None) -> str:
    """Run ``scripts/runpod_deploy_worker.sh``; return API base URL (no trailing slash)."""
    root = repo_root or str(REPO_ROOT)
    script = os.path.join(root, 'scripts', 'runpod_deploy_worker.sh')
    if not os.path.isfile(script):
        raise FileNotFoundError(script)
    env = os.environ.copy()
    if pod_name:
        env['POD_NAME'] = pod_name.strip()
        print(f'  POD_NAME={pod_name.strip()}')
    print(f'Running: {script} (cwd={root}) …')
    p = subprocess.run(['bash', script], cwd=root, capture_output=True, text=True, timeout=timeout_sec, env=env)
    blob = (p.stdout or '') + '\n' + (p.stderr or '')
    tail = blob[-12000:] if len(blob) > 12000 else blob
    print(tail)
    if p.returncode != 0:
        raise RuntimeError(f'Deploy script failed (exit {p.returncode})')
    pod_id = parse_runpod_pod_id_from_deploy_output(blob)
    base = f'https://{pod_id}-8000.proxy.runpod.net'
    print(f'\n→ LACUNA_DATA_API_BASE={base}')
    return base

def ensure_classify_runpod_bases(*, pod_count: int, stem: str | None=None, deploy: bool, use_graphql: bool) -> list[str]:
    """Ensure up to ``pod_count`` worker base URLs; set ``LACUNA_DATA_API_BASE``.

    When ``LACUNA_DATA_CLASSIFY_ASYNC_DEPLOY`` is on (default) and ``deploy`` is True,
    returns as soon as at least one worker is healthy and starts a background slot
    reconciler (unless ``LACUNA_DATA_CLASSIFY_SLOT_RECONCILE=0``) that refreshes
    GraphQL and retries deploy with backoff until every slot is filled. Each poll also
    aligns slot/rotation URLs with the API when proxy ids change (``LACUNA_DATA_CLASSIFY_GRAPHQL_URL_SYNC``).
    New URLs are appended to the shared list (and env) when ready so ``batch_classify_runpod``
    picks them up immediately for round-robin and higher concurrency.

    When async deploy is off, uses the legacy blocking deploy loop. If at least one
    worker exists but not all slots are filled, a warning is printed and the run
    continues with available URLs only.
    """
    _runpod_clear_shared_bases_for_classify()
    n = max(1, int(pod_count))
    stem = (stem or os.environ.get('POD_NAME') or 'lacuna-data-worker').strip()
    raw = (os.environ.get('LACUNA_DATA_API_BASE') or '').strip()
    from_env = _parse_runpod_classifier_bases(raw)
    if from_env:
        stale = [b for b in from_env if data_worker_base_is_stale_404(b)]
        if stale:
            print('One or more LACUNA_DATA_API_BASE URLs look dead (404 on /health); clearing to rediscover/redeploy.')
            os.environ.pop('LACUNA_DATA_API_BASE', None)
            from_env = []
    slots: list[str | None] = [None] * n
    for i in range(min(n, len(from_env))):
        slots[i] = from_env[i]
        print(f'  Slot {i + 1}/{n}: reuse from LACUNA_DATA_API_BASE → {from_env[i]}')
    rk = (os.environ.get('RUNPOD_API_KEY') or '').strip()
    _discovery_debug_shown = False

    def fill_from_graphql() -> None:
        nonlocal slots, _discovery_debug_shown
        if not rk or not use_graphql:
            return
        try:
            found = discover_classify_bases(rk, stem, n)
        except Exception as e:
            print(f'RunPod GraphQL discover_classify_bases failed ({e!r})')
            return
        for idx in range(1, n + 1):
            if slots[idx - 1] is None and idx in found:
                slots[idx - 1] = found[idx]
                print(f'  Slot {idx}/{n}: reuse existing pod (RunPod API) → {found[idx]}')
        if _runpod_graphql_url_sync_enabled():
            for idx in range(1, n + 1):
                if idx not in found:
                    continue
                cur = slots[idx - 1]
                if not cur:
                    continue
                want = found[idx]
                if _normalize_runpod_base_url(cur) == _normalize_runpod_base_url(want):
                    continue
                print(f'  Slot {idx}/{n}: RunPod API proxy URL changed ({cur!r} → {want!r}); syncing')
                _runpod_replace_rotation_base_url(cur, want)
                slots[idx - 1] = want.strip().rstrip('/')
        if len(found) < n and (not _discovery_debug_shown) and any((slots[i] is None for i in range(n))):
            _discovery_debug_shown = True
            warn_if_classify_slots_missing(rk, stem, n, found)
    print(f'Classify cluster: target {n} worker(s), stem={stem!r}')
    fill_from_graphql()
    async_on = _runpod_classify_async_deploy_enabled() and deploy
    if async_on:
        for i in range(n):
            if slots[i] is not None:
                wait_for_data_worker(slots[i])
        out = [slots[i] for i in range(n) if slots[i]]
        missing = [idx for idx in range(1, n + 1) if slots[idx - 1] is None]
        if not out:
            if not missing:
                raise RuntimeError('No classify worker slots to deploy.')
            idx0 = missing[0]
            pname = classify_worker_pod_name(stem, n, idx0)
            print(f'\n  Slot {idx0}/{n}: no workers yet — deploying {pname!r} (blocking until healthy) …')
            slots[idx0 - 1] = deploy_data_worker_pod(pod_name=pname)
            wait_for_data_worker(slots[idx0 - 1])
            fill_from_graphql()
            out = [slots[i] for i in range(n) if slots[i]]
            missing = [idx for idx in range(1, n + 1) if slots[idx - 1] is None]
        if not out:
            raise RuntimeError('No healthy classify workers after bootstrap deploy.')
        still_missing = [idx for idx in range(1, n + 1) if slots[idx - 1] is None]
        if not still_missing:
            fill_from_graphql()
            out = [slots[i] for i in range(n) if slots[i]]
            os.environ['LACUNA_DATA_API_BASE'] = ','.join(out)
            print(f'\n→ LACUNA_DATA_API_BASE={os.environ['LACUNA_DATA_API_BASE']}')
            return out
        shared: list[str] = list(out)
        lock = threading.Lock()

        def _append_worker_url(url: str) -> None:
            u = url.rstrip('/')
            k = _normalize_runpod_base_url(u)
            with lock:
                for b in shared:
                    if _normalize_runpod_base_url(b) == k:
                        return
                shared.append(u)
                _runpod_maintain_shared_bases_unlocked(shared)
            print(f'  [async deploy] Worker ready → {u} (now {_len_shared_bases()} URL(s) in rotation)')
        os.environ['LACUNA_DATA_API_BASE'] = ','.join(shared)
        _runpod_attach_shared_bases_for_classify(shared, lock)
        _runpod_register_classify_slot_tracking(slots, n, fill_from_graphql)
        _runpod_maintain_shared_bases()
        if _runpod_slot_reconcile_enabled():
            _runpod_start_slot_reconciler(stem=stem, n=n, slots=slots, shared=shared, lock=lock, fill_from_graphql=fill_from_graphql, append_worker_url=_append_worker_url)
            extra_msg = f'{len(still_missing)} more slot(s) filled by reconcile loop during batch (set LACUNA_DATA_CLASSIFY_SLOT_RECONCILE=0 for one-shot threads only).'
        else:

            def _deploy_slot_bg(idx: int) -> None:
                pname = classify_worker_pod_name(stem, n, idx)
                try:
                    print(f'\n  [background] Slot {idx}/{n}: deploying {pname!r} …')
                    base_url = deploy_data_worker_pod(pod_name=pname)
                    wait_for_data_worker(base_url)
                    fill_from_graphql()
                    if slots[idx - 1] is None:
                        slots[idx - 1] = base_url
                    _append_worker_url(base_url)
                except Exception as e:
                    print(f'  [background] Slot {idx}/{n} ({pname!r}) failed: {e!r}')
            for idx in still_missing:
                threading.Thread(target=_deploy_slot_bg, args=(idx,), daemon=True, name=f'runpod-classify-deploy-{idx}').start()
            extra_msg = f'{len(still_missing)} more deploying in background (one-shot threads; no retry loop).'
        print(f'\n  Async deploy: starting classify with {len(shared)}/{n} healthy worker(s); {extra_msg} (LACUNA_DATA_CLASSIFY_ASYNC_DEPLOY=1).')
        return shared
    for idx in range(1, n + 1):
        fill_from_graphql()
        if slots[idx - 1] is not None:
            continue
        if not deploy:
            if any((slots[i] is not None for i in range(n))):
                print(f'WARNING: missing slot {idx}/{n} ({classify_worker_pod_name(stem, n, idx)!r}); continuing with available worker(s) only.')
                break
            raise RuntimeError(f'Missing classify worker slot {idx}/{n} ({classify_worker_pod_name(stem, n, idx)!r}). Set DEPLOY_RUNPOD_FOR_CLASSIFY=True, or set LACUNA_DATA_API_BASE with enough URLs, or start a pod with that name and re-run.')
        pname = classify_worker_pod_name(stem, n, idx)
        print(f'\n  Slot {idx}/{n}: no existing pod — deploying {pname!r} …')
        slots[idx - 1] = deploy_data_worker_pod(pod_name=pname)
        wait_for_data_worker(slots[idx - 1])
        fill_from_graphql()
    out = [s for s in slots if s]
    if not out:
        raise RuntimeError('No classify worker base URLs available.')
    if len(out) < n:
        print(f'WARNING: only {len(out)}/{n} worker URL(s) — pipeline will use what is available.')
    print('Waiting for /health on all workers …')
    for b in out:
        wait_for_data_worker(b)
    os.environ['LACUNA_DATA_API_BASE'] = ','.join(out)
    print(f'\n→ LACUNA_DATA_API_BASE={os.environ['LACUNA_DATA_API_BASE']}')
    return out

def wait_for_data_worker(base_url: str, *, timeout_sec: float=900.0, interval_sec: float=10.0, api_key: str | None=None) -> dict:
    """Poll ``GET /health`` until the worker responds (pod may still be installing).

    Uses a browser-like User-Agent because RunPod's proxy often returns 403 to
    ``Python-urllib``. Does not send ``X-API-Key`` on ``/health`` (that route is public).
    """
    _ = api_key
    base_url = base_url.rstrip('/')
    deadline = time.time() + timeout_sec
    last_err: str | None = None
    attempt = 0
    http404_count = 0
    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(base_url + '/health', method='GET')
            for hk, hv in _RUNPOD_HTTP_HEADERS.items():
                req.add_header(hk, hv)
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data.get('status') == 'ok':
                print(f'Worker healthy: {data}')
                return data
        except urllib.error.HTTPError as e:
            last_err = repr(e)
            if getattr(e, 'code', None) == 404:
                http404_count += 1
            print(f'Waiting for worker… ({last_err})')
        except Exception as e:
            last_err = repr(e)
            print(f'Waiting for worker… ({last_err})')
        if http404_count == 3 or (http404_count > 0 and attempt == 10):
            print('  Hint: repeated HTTP 404 usually means nothing is listening on port 8000 yet (first boot: git clone + pip + optional model preload can take 15–40+ minutes) or the startup script failed. On the pod, run:\n    tail -n 80 /workspace/data-worker-api.log\n    tail -n 80 /workspace/jupyter.log\n    ls -la /workspace && ls -la /workspace/*/* 2>/dev/null | head\n  RunPod web console → Pod → Connect (web terminal) or: ssh <pod_id>@ssh.runpod.io')
        time.sleep(interval_sec)
    raise TimeoutError(f'No healthy /health from {base_url} within {timeout_sec}s (last: {last_err})')

def data_worker_base_is_stale_404(base_url: str, *, attempts: int=4, interval_sec: float=3.0, request_timeout: float=20.0) -> bool:
    """Return True if ``GET /health`` only returns HTTP 404 (common when the pod URL is dead).

    Returns False if the worker responds ok, or if errors are not all 404 (502 during boot,
    timeouts, etc.) — in those cases ``wait_for_data_worker`` should decide.
    """
    base_url = base_url.rstrip('/')
    url = base_url + '/health'
    saw_non_404 = False
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, method='GET')
            for hk, hv in _RUNPOD_HTTP_HEADERS.items():
                req.add_header(hk, hv)
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data.get('status') == 'ok':
                return False
            saw_non_404 = True
        except urllib.error.HTTPError as e:
            if e.code != 404:
                saw_non_404 = True
        except Exception:
            saw_non_404 = True
        if i + 1 < attempts:
            time.sleep(interval_sec)
    return not saw_non_404

def _load_classify_state_file(state_path: str, *, context: str) -> tuple[dict, bool]:
    """Load ``*.state.json`` for classify; if corrupt or truncated, move aside and return fresh.

    Returns ``(state, recovered_from_corrupt)``. The second flag is True when a damaged file was
    quarantined — callers should avoid wiping ``category_dir`` shards in that case.
    """
    empty: dict = {'completed_batches': [], 'results': {}, 'pending_batch_ids': []}
    if not os.path.exists(state_path):
        return ({**empty}, False)
    try:
        with open(state_path, 'r', encoding='utf-8') as sf:
            raw = sf.read()
        if not raw.strip():
            raise json.JSONDecodeError('empty state file', raw, 0)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError('state root must be a JSON object')
        out = {**empty, **data}
        out.setdefault('completed_batches', [])
        out.setdefault('results', {})
        out.setdefault('pending_batch_ids', [])
        if not isinstance(out['results'], dict):
            out['results'] = {}
        return (out, False)
    except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as e:
        bak = f'{state_path}.corrupt.{int(time.time())}.bak'
        try:
            os.replace(state_path, bak)
        except OSError:
            try:
                shutil.copy2(state_path, bak)
                os.remove(state_path)
            except OSError as e2:
                print(f'WARNING: could not quarantine corrupt state file: {e2!r}')
                return ({**empty}, False)
        print(f'WARNING: {context}: unreadable state at {state_path!r} ({e!r}). Quarantined as {bak!r}; continuing with empty state (re-classify or merge from backup).')
        return ({**empty}, True)

def _atomic_json_dump_state(path: str, state: dict, **dump_kw) -> None:
    """Atomically write classify state (crash-safe vs direct json.dump to path)."""
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix='.tmp', prefix=os.path.basename(path) + '.', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as wf:
            json.dump(state, wf, **dump_kw)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def _classify_batch_http_error_message(code: int, detail: str) -> str:
    """Append FastAPI/uvicorn JSON ``detail`` (worker classify errors) for easier debugging."""
    base = f'classify/batch HTTP {code}: {detail}'
    if code != 500 or not (detail or '').strip():
        return base
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return base
    inner = parsed.get('detail')
    if isinstance(inner, dict):
        parts: list[str] = []
        if inner.get('error'):
            parts.append(str(inner['error']))
        if inner.get('message'):
            parts.append(str(inner['message']))
        if 'item_index' in inner:
            parts.append(f'item_index={inner['item_index']}')
        if parts:
            return base + ' | ' + ' | '.join(parts)
    elif isinstance(inner, str) and inner.strip():
        return base + ' | ' + inner.strip()
    return base

def _print_worker_500_detail_if_json(detail: str) -> None:
    try:
        p = json.loads(detail)
    except json.JSONDecodeError:
        return
    d = p.get('detail')
    if not isinstance(d, dict):
        return
    if not (d.get('message') or d.get('error')):
        return
    suf = f' [item {d['item_index']}]' if isinstance(d.get('item_index'), int) else ''
    print(f'    └ {d.get('error', 'Error')}: {d.get('message', '')}{suf}')
_CLASSIFY_BATCH_FAILURE_LOG_LOCK = threading.Lock()

def _classify_http_body_fatal_worker_gpu(http_detail: str) -> bool:
    """True when HTTP body indicates an unrecoverable CUDA/GPU worker state."""
    if not (http_detail or '').strip():
        return False
    low = http_detail.lower()
    needles = ('device-side assert', 'cuda error: device-side assert', 'an illegal memory access', 'cuda error: an illegal memory access was encountered')
    return any((n in low for n in needles))

def _runpod_fatal_cuda_replace_enabled() -> bool:
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_FATAL_CUDA_REPLACE', '1').strip().lower()
    return raw not in ('0', 'false', 'no', 'off', 'disabled')

def _len_shared_bases() -> int:
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is None:
        return 0
    if bl is not None:
        with bl:
            return len(sb)
    return len(sb)

def _runpod_append_base_to_shared(new_url: str) -> None:
    u = new_url.rstrip('/')
    k = _normalize_runpod_base_url(u)
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is None:
        return
    if bl is not None:
        with bl:
            if not any((_normalize_runpod_base_url(b) == k for b in sb)):
                sb.append(u)
            _runpod_maintain_shared_bases_unlocked(sb)
    else:
        if not any((_normalize_runpod_base_url(b) == k for b in sb)):
            sb.append(u)
        _runpod_maintain_shared_bases_unlocked(sb)

def _runpod_redeploy_worker_background(pod_name: str) -> None:

    def _job() -> None:
        global _runpod_redeploy_names_inflight
        with _runpod_redeploy_lock:
            if pod_name in _runpod_redeploy_names_inflight:
                return
            _runpod_redeploy_names_inflight.add(pod_name)
        try:
            print(f'\n  [replace worker] background deploy {pod_name!r} …')
            nb = deploy_data_worker_pod(pod_name=pod_name)
            rk = (os.environ.get('RUNPOD_API_KEY') or '').strip()
            wait_for_data_worker(nb, api_key=rk or None)
            _runpod_append_base_to_shared(nb)
            print(f'  [replace worker] {pod_name!r} ready → {nb.rstrip('/')} (rotation size {_len_shared_bases()})')
        except Exception as e:
            print(f'  WARNING: [replace worker] deploy failed for {pod_name!r}: {e!r}')
        finally:
            with _runpod_redeploy_lock:
                _runpod_redeploy_names_inflight.discard(pod_name)
    threading.Thread(target=_job, daemon=True, name=f'runpod-fatal-replace-{pod_name}').start()

def _runpod_force_drop_fatal_cuda_worker(bases: list[str], failed_base: str, worker_api_key: str | None, *, bases_lock: threading.Lock | None=None) -> None:
    """Remove URL from rotation, stop+terminate the pod, redeploy same pod name."""
    key = _normalize_runpod_base_url(failed_base)
    _ = worker_api_key
    rk = os.environ.get('RUNPOD_API_KEY', '').strip()
    pid = pod_id_from_worker_proxy_base(failed_base)
    pname = pod_display_name_for_id(rk, pid) if rk and pid else None
    if not pname and rk and pid:
        time.sleep(1.5)
        pname = pod_display_name_for_id(rk, pid)

    def _strip_from_bases() -> int:

        def _inner() -> int:
            bases[:] = [b for b in bases if _normalize_runpod_base_url(b) != key]
            return len(bases)
        if bases_lock is not None:
            with bases_lock:
                return _inner()
        return _inner()
    remaining = _strip_from_bases()
    _runpod_clear_slots_matching_base(key)
    resync = _RUNPOD_CLASSIFY_GRAPHQL_RESYNC
    if resync is not None:
        try:
            resync()
        except Exception as e:
            print(f'  WARNING: GraphQL resync after fatal CUDA: {e!r}')
    _runpod_maintain_shared_bases()
    with _runpod_base_failures_lock:
        _runpod_base_consecutive_failures.pop(key, None)
    print(f'  Fatal GPU/worker error — removed {failed_base!r} from rotation ({remaining} base URL(s) left). RunPod pod id={pid!r} name={pname!r}.')
    if rk and pid:
        try:
            stop_then_terminate_pod(rk, pid)
            print(f'  RunPod stop→terminate ok for pod {pid!r} ({pname!r}).')
        except Exception as e:
            print(f'WARNING: RunPod stop/terminate failed for {failed_base!r}: {e!r}')
    elif not rk:
        print('WARNING: RUNPOD_API_KEY unset — cannot stop/terminate the bad pod from here.')
    if not _runpod_fatal_cuda_replace_enabled() or not pname:
        if not pname and _runpod_fatal_cuda_replace_enabled():
            print('WARNING: could not resolve pod name for redeploy — deploy a replacement manually with the same POD_NAME/stem.')
        return
    if remaining == 0:
        print(f'\n  [replace worker] blocking deploy {pname!r} (pool was empty) …')
        nb = deploy_data_worker_pod(pod_name=pname)
        rk2 = (os.environ.get('RUNPOD_API_KEY') or '').strip()
        wait_for_data_worker(nb, api_key=rk2 or None)
        _runpod_append_base_to_shared(nb)
        print(f'  [replace worker] {pname!r} ready (rotation size {_len_shared_bases()}).')
    else:
        _runpod_redeploy_worker_background(pname)

def _classify_batch_failure_log_path_resolved(state_path: str) -> str | None:
    """Append-only JSONL for failed classify/batch HTTP calls. None = disabled.

    Env ``LACUNA_DATA_CLASSIFY_FAILURE_LOG``: path, or ``0``/``false``/``off`` to disable.
    Default: ``{state_path}.classify_batch_failures.jsonl``.
    """
    raw = os.environ.get('LACUNA_DATA_CLASSIFY_FAILURE_LOG', '').strip()
    if raw.lower() in ('0', 'false', 'no', 'off', 'disabled', 'none'):
        return None
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    return os.path.abspath(state_path + '.classify_batch_failures.jsonl')

def _append_classify_batch_failure_log(log_path: str, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + '\n'
    d = os.path.dirname(os.path.abspath(log_path)) or '.'
    os.makedirs(d, exist_ok=True)
    with _CLASSIFY_BATCH_FAILURE_LOG_LOCK:
        with open(log_path, 'a', encoding='utf-8') as wf:
            wf.write(line)
    print(f'  Logged batch failure → {log_path}')

def _post_classify_batch_http(base_url: str, items: list[dict], api_key: str | None, timeout: float, *, sleep_after_error: float=5.0, max_retries: int=12, failure_log_path: str | None=None, failure_extra: dict | None=None) -> list[dict]:
    url = base_url.rstrip('/') + '/classify/batch'
    payload = json.dumps({'items': items}).encode('utf-8')
    last_exc: Exception | None = None

    def _flush_batch_failure(msg: str, *, kind: str, http_code: int | None=None, http_detail: str | None=None) -> None:
        if not failure_log_path:
            return
        rec: dict = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'unix_time': time.time(), 'base_url': base_url, 'error_kind': kind, 'message': msg, 'items': items}
        if failure_extra:
            for k in ('input_path', 'line_indices', 'samples'):
                if k in failure_extra:
                    rec[k] = failure_extra[k]
        if http_code is not None:
            rec['http_code'] = http_code
        if http_detail is not None:
            lim = 16384
            body = http_detail if len(http_detail) <= lim else http_detail[:lim] + '...(truncated)'
            rec['http_body'] = body
            if http_code == 500:
                try:
                    p = json.loads(http_detail)
                    d = p.get('detail')
                    if isinstance(d, dict):
                        rec['worker_detail'] = d
                except json.JSONDecodeError:
                    pass
        _append_classify_batch_failure_log(failure_log_path, rec)
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload, method='POST')
        req.add_header('Content-Type', 'application/json')
        for hk, hv in _RUNPOD_HTTP_HEADERS.items():
            req.add_header(hk, hv)
        if api_key:
            req.add_header('X-API-Key', api_key)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            return body['results']
        except urllib.error.HTTPError as e:
            last_exc = e
            try:
                detail = e.read().decode()
            except Exception:
                detail = ''
            try:
                code = int(e.code)
            except (TypeError, ValueError):
                code = 0
            if code == 500 and _classify_http_body_fatal_worker_gpu(detail):
                _print_worker_500_detail_if_json(detail)
                print('  classify/batch HTTP 500: fatal GPU/worker error — not retrying this pod (drop + stop/terminate + replace).')
                msg = _classify_batch_http_error_message(code, detail) + CLASSIFY_FATAL_CUDA_MARKER
                _flush_batch_failure(msg, kind='http_fatal_cuda', http_code=code, http_detail=detail)
                raise RuntimeError(msg) from e
            if code == 503 and _runpod_classify_busy_reroute_enabled():
                msg = _classify_batch_http_error_message(code, detail)
                _flush_batch_failure(msg, kind='http_busy', http_code=code, http_detail=detail)
                raise RuntimeError(msg + CLASSIFY_POD_BUSY_MARKER) from e
            retriable = code in (403, 404, 500, 502, 503, 504)
            if retriable and attempt < max_retries - 1:
                delay = sleep_after_error
                if code in (404, 500, 502, 503, 504):
                    delay = min(90.0, sleep_after_error * 2 ** min(attempt, 5))
                hint = ' (pod/proxy may be down; retrying)' if code == 404 else ''
                if code == 500:
                    hint = ' (worker internal error; retrying)'
                    _print_worker_500_detail_if_json(detail)
                print(f'  classify/batch HTTP {code}, attempt {attempt + 1}/{max_retries}, sleep {delay:.1f}s…{hint}')
                time.sleep(delay)
                continue
            msg = _classify_batch_http_error_message(code, detail)
            if code == 404:
                msg += ' | Often: RunPod pod stopped/OOM/preempt, or stale LACUNA_DATA_API_BASE. Check the pod in RunPod console, fix the base URL in .env, then re-run (classification state resumes from disk).'
            _flush_batch_failure(msg, kind='http', http_code=code, http_detail=detail)
            raise RuntimeError(msg) from e
        except urllib.error.URLError as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = min(90.0, sleep_after_error * 2 ** min(attempt, 5))
                print(f'  classify/batch URLError ({e}), attempt {attempt + 1}/{max_retries}, sleep {delay:.1f}s…')
                time.sleep(delay)
                continue
            _flush_batch_failure(str(e), kind='url')
            raise
    _flush_batch_failure(str(last_exc), kind='exhausted_retries')
    raise RuntimeError(str(last_exc))

def _post_classify_batch_http_resilient(base_url_holder: list[str], items: list[dict], api_key: str | None, timeout: float, *, sleep_after_error: float=5.0, max_http_retries: int=12, max_base_refreshes: int=4, failure_log_path: str | None=None, failure_extra: dict | None=None) -> list[dict]:
    """POST classify/batch; after HTTP retries exhaust on 404, refresh proxy URL via RunPod GraphQL and retry."""
    refreshes = 0
    while True:
        try:
            return _post_classify_batch_http(base_url_holder[0], items, api_key, timeout, sleep_after_error=sleep_after_error, max_retries=max_http_retries, failure_log_path=failure_log_path, failure_extra=failure_extra)
        except RuntimeError as exc:
            msg = str(exc)
            if 'HTTP 404' not in msg:
                raise
            if refreshes >= max_base_refreshes:
                raise RuntimeError(f'{msg} | Giving up after {max_base_refreshes} GraphQL base-URL refresh(es).') from exc
            rk = os.environ.get('RUNPOD_API_KEY', '').strip()
            if not rk:
                raise RuntimeError(f'{msg} | Set RUNPOD_API_KEY to auto-refresh worker URL after 404.') from exc
            nb = discover_worker_base(rk)
            old = base_url_holder[0].rstrip('/')
            if not nb:
                raise RuntimeError(f'{msg} | No active pod matched POD_NAME in RunPod API; redeploy or fix LACUNA_DATA_API_BASE.') from exc
            nb = nb.rstrip('/')
            if nb == old:
                raise RuntimeError(f'{msg} | GraphQL returned the same URL; restart/redeploy the pod in RunPod console.') from exc
            refreshes += 1
            base_url_holder[0] = nb
            os.environ['LACUNA_DATA_API_BASE'] = nb
            print(f'  classify/batch HTTP 404: refreshed LACUNA_DATA_API_BASE → {nb} (GraphQL refresh {refreshes}/{max_base_refreshes}); waiting for /health…')
            wait_for_data_worker(nb, api_key=api_key)

def _write_output_from_disk(input_path: str, output_path: str, results: dict, total: int) -> None:
    """Write classified JSONL using the same line indices as ``readlines()``-based path."""
    with open(input_path, 'r', encoding='utf-8') as rf, open(output_path, 'w', encoding='utf-8') as wf:
        for i in range(total):
            raw = rf.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            sample = json.loads(line)
            labs = _normalize_stored_labels(results.get(str(i)))
            sample['categories'] = labs
            sample['category'] = ','.join(labs)
            wf.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'  Output written: {output_path}')

def _clear_category_jsonl(category_dir: str) -> None:
    """Remove ``*.jsonl`` shards under ``category_dir``."""
    if not os.path.isdir(category_dir):
        return
    for fn in os.listdir(category_dir):
        if fn.endswith('.jsonl'):
            os.remove(os.path.join(category_dir, fn))

def _record_for_category_shard(sample: dict) -> dict:
    """Shallow copy without ``category`` (HF rows may already include it)."""
    return {k: v for k, v in sample.items() if k != 'category'}

def _append_batch_to_category_dir(category_dir: str, batch_samples: list[dict], labels_per_row: list[list[str]]) -> None:
    """Append each sample to one shard per label (multi-label); no ``category`` field."""
    os.makedirs(category_dir, exist_ok=True)
    by_cat: dict[str, list[str]] = {}
    for sample, labs in zip(batch_samples, labels_per_row):
        for lab in _normalize_stored_labels(labs):
            line = json.dumps(_record_for_category_shard(sample), ensure_ascii=False) + '\n'
            by_cat.setdefault(lab, []).append(line)
    for cat, lines in by_cat.items():
        p = os.path.join(category_dir, f'{cat}.jsonl')
        with open(p, 'a', encoding='utf-8') as wf:
            wf.writelines(lines)

def _sync_category_dir_from_state(input_path: str, total: int, results: dict, category_dir: str) -> None:
    """Rebuild per-category JSONL from input + state (no merged file)."""
    _clear_category_jsonl(category_dir)
    os.makedirs(category_dir, exist_ok=True)
    by_cat: dict[str, list[str]] = {c: [] for c in VALID_CATEGORIES}
    with open(input_path, 'r', encoding='utf-8') as rf:
        for i in range(total):
            raw = rf.readline()
            if not raw:
                break
            line = raw.strip()
            if not line:
                continue
            labs = _normalize_stored_labels(results.get(str(i)))
            sample = json.loads(line)
            sline = json.dumps(_record_for_category_shard(sample), ensure_ascii=False) + '\n'
            for lab in labs:
                by_cat[lab].append(sline)
    for cat, lines in by_cat.items():
        if not lines:
            continue
        p = os.path.join(category_dir, f'{cat}.jsonl')
        with open(p, 'w', encoding='utf-8') as wf:
            wf.writelines(lines)

def _batch_classify_runpod_stream(input_path: str, output_path: str, *, base_url: str | None=None, api_key: str | None=None, state_path: str | None=None, batch_size: int=RUNPOD_CLASSIFY_BATCH_SIZE, max_samples: int | None=None, request_timeout: float=900.0, category_dir: str | None=None, clear_category_dir: bool=True, parallel_workers: int | None=None) -> str:
    """Stream JSONL from disk (for multi-GB files); same state file format as non-streaming.

    If ``category_dir`` is set, each batch appends originals to ``{identity,values,persona,general}.jsonl``
    there (no ``category`` field). The merged ``output_path`` JSONL is not written.

    Set ``LACUNA_DATA_API_BASE`` to one URL or several separated by commas / whitespace to spread
    load across RunPod workers. ``parallel_workers`` or env ``LACUNA_DATA_CLASSIFY_PARALLEL`` caps
    concurrent in-flight HTTP batches (single local writer for ``.state.json`` and category appends).
    ``LACUNA_DATA_CLASSIFY_MAX_IN_FLIGHT_PER_BASE`` (default 1) avoids stacking multiple batches on one URL;
    HTTP 503 can reroute to another pod (``LACUNA_DATA_CLASSIFY_BUSY_REROUTE``).
    With a single base URL, GraphQL base refresh after HTTP 404 is used only when running
    sequentially (one in-flight batch).
    """
    batch_size = _runpod_classify_batch_size_resolved(batch_size)
    raw_bases = (base_url or os.environ.get('LACUNA_DATA_API_BASE', '')).strip()
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is not None:
        bases = sb
    else:
        bases = _parse_runpod_classifier_bases(raw_bases)
        if bases:
            _runpod_attach_shared_bases_for_classify(bases, threading.Lock())
        sb = _RUNPOD_CLASSIFY_SHARED_BASES
        bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
        if sb is not None:
            bases = sb
    if not bases:
        raise ValueError('Set LACUNA_DATA_API_BASE in .env (e.g. https://YOUR_POD-8000.proxy.runpod.net) or pass base_url=')
    try:
        if api_key is None:
            k = os.environ.get('LACUNA_DATA_API_KEY', '').strip()
            api_key = k or None
        if state_path is None:
            state_path = output_path + '.state.json'
        failure_log_path = _classify_batch_failure_log_path_resolved(state_path)
        with open(input_path, 'r', encoding='utf-8') as cf:
            file_lines = sum((1 for _ in cf))
        total = file_lines if max_samples is None else min(file_lines, max_samples)
        state, _state_corrupt_recover = _load_classify_state_file(state_path, context='RunPod stream classify')
        if state.get('results'):
            print(f'Resuming: {len(state['results']):,}/{total:,} already in state')
        state.setdefault('pending_batch_ids', [])
        if state['pending_batch_ids']:
            print('WARNING: Clearing pending_batch_ids from a prior non-RunPod batch run — not used with RunPod HTTP classification.')
            state['pending_batch_ids'] = []
            _atomic_json_dump_state(state_path, state)
        if category_dir and clear_category_dir and (not state.get('results')) and (not _state_corrupt_recover):
            _clear_category_jsonl(category_dir)
            print(f'  Cleared *.jsonl under {category_dir} (fresh run)')
        if len(state['results']) >= total:
            print(f'Already complete: {total:,}/{total:,}')
            if category_dir:
                _sync_category_dir_from_state(input_path, total, state['results'], category_dir)
                print(f'  Category files synced under {category_dir}')
            else:
                _write_output_from_disk(input_path, output_path, state['results'], total)
            return output_path
        target_n = max(1, int(os.environ.get('LACUNA_DATA_CLASSIFY_POD_COUNT', str(max(len(bases), 1)))))
        parallel_w_cap = _runpod_classify_parallel_workers(target_n, parallel_workers)

        def _effective_parallel_w() -> int:
            if bl is not None:
                with bl:
                    nb = len(bases)
            else:
                nb = len(bases)
            return min(parallel_w_cap, max(1, nb))

        def _bases_nonempty() -> bool:
            if bl is not None:
                with bl:
                    return bool(bases)
            return bool(bases)
        use_graphql_refresh = len(bases) == 1 and parallel_w_cap == 1
        base_holder = [bases[0]] if use_graphql_refresh else []
        print(f'RunPod classify: {len(bases):,} base URL(s) (more may join while async deploy finishes), up to {parallel_w_cap:,} concurrent batch(es) in flight' + (' (GraphQL URL refresh enabled)' if use_graphql_refresh else ''))
        _dbd = _runpod_bad_base_drop_threshold()
        if _dbd > 0:
            print(f'  Bad-base drop: after {_dbd} consecutive batch failure(s) on one URL, it is removed briefly; failed batches re-queue on others. Dropped-pod handling={_runpod_dropped_pod_handling()!r} (env LACUNA_DATA_CLASSIFY_DROPPED_POD_HANDLING=restart|stop|none). Set LACUNA_DATA_CLASSIFY_DROP_BAD_BASE_AFTER=0 to disable removal.')
        already_done = {int(k) for k in state['results'].keys()}
        start_time = time.time()
        batch_indices: list[int] = []
        batch_samples: list[dict] = []
        chunk_i = 0
        state_lock = threading.Lock()
        rr = 0
        qmax_res = _runpod_classify_pending_batch_queue_max_resolved(parallel_w_cap)
        batch_queue: deque | None = deque() if qmax_res > 0 else None
        _mip_cap = _runpod_classify_max_in_flight_per_base()
        if _mip_cap > 0:
            print(f'  Per-base in-flight cap: {_mip_cap} (LACUNA_DATA_CLASSIFY_MAX_IN_FLIGHT_PER_BASE; 0=unlimited legacy)')
        if _runpod_classify_busy_reroute_enabled():
            print('  HTTP 503 → reroute to another pod (LACUNA_DATA_CLASSIFY_BUSY_REROUTE=1; set 0 to retry same URL)')
        if qmax_res > 0:
            print(f'  Pending batch queue max: {qmax_res} (LACUNA_DATA_CLASSIFY_PENDING_BATCH_QUEUE_MAX)')

        def _pick_base_rr() -> str:
            nonlocal rr
            if bl is not None:
                with bl:
                    if not bases:
                        raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                    b_pick = bases[rr % len(bases)]
                    rr += 1
            else:
                if not bases:
                    raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                b_pick = bases[rr % len(bases)]
                rr += 1
            return b_pick

        def _count_inflight_by_norm(pr: dict) -> dict[str, int]:
            out: dict[str, int] = {}
            for _fut, (_bi, _bs, bu) in pr.items():
                k = _normalize_runpod_base_url(bu)
                out[k] = out.get(k, 0) + 1
            return out

        def _run_http_batch(target_base: str, samples: list[dict], line_indices: list[int]) -> list[dict]:
            items = [{'text': _sample_to_preview(s)} for s in samples]
            extra = {'input_path': input_path, 'line_indices': line_indices, 'samples': samples}
            if use_graphql_refresh:
                return _post_classify_batch_http_resilient(base_holder, items, api_key, request_timeout, failure_log_path=failure_log_path, failure_extra=extra)
            return _post_classify_batch_http(target_base, items, api_key, request_timeout, failure_log_path=failure_log_path, failure_extra=extra)

        def _run_http_batch_wrapped(target_base: str, samples: list[dict], line_indices: list[int]) -> tuple[str, object]:
            try:
                return ('ok', _run_http_batch(target_base, samples, line_indices))
            except BaseException as e:
                return ('err', e)

        def _apply_batch_result(bi: list[int], bs: list[dict], http_results: list[dict], b_used: str) -> None:
            nonlocal chunk_i
            if len(http_results) != len(bi):
                raise RuntimeError(f'Expected {len(bi)} results, got {len(http_results)}')
            labels_per_row: list[list[str]] = []
            for idx, row, sample in zip(bi, http_results, bs):
                labs = _labels_from_classify_row(row, sample=sample)
                labels_per_row.append(labs)
            with state_lock:
                for idx, labs in zip(bi, labels_per_row):
                    state['results'][str(idx)] = labs
                chunk_i += 1
                bn = chunk_i
                state['completed_batches'].append(f'runpod:stream:{bn}')
                _atomic_json_dump_state(state_path, state)
                print(f'  Checkpoint {len(state['results']):,} lines with labels (stream batch #{bn}, via {b_used})')
                if category_dir:
                    _append_batch_to_category_dir(category_dir, bs, labels_per_row)
                    print(f'  Appended batch to category files under {category_dir}')

        def _drain_one_future(pending: dict, ex: ThreadPoolExecutor) -> None:
            done_set, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                bi, bs, b_used = pending.pop(fut)
                tag, payload = fut.result()
                if tag == 'ok':
                    _runpod_record_batch_success(b_used)
                    _apply_batch_result(bi, bs, payload, b_used)
                else:
                    exc = payload
                    if CLASSIFY_FATAL_CUDA_MARKER in str(exc):
                        print(f'  Batch failed on {b_used!r} (fatal GPU/worker): {exc!r}')
                        _runpod_force_drop_fatal_cuda_worker(bases, b_used, api_key, bases_lock=bl)
                        nfail, _dropped = (1, True)
                    elif CLASSIFY_POD_BUSY_MARKER in str(exc):
                        print(f'  Pod busy (HTTP 503) on {b_used!r} — rerouting: {exc!r}')
                        nfail, _dropped = (0, False)
                    else:
                        nfail, _dropped = _runpod_record_batch_failure_and_maybe_drop(bases, b_used, worker_api_key=api_key, bases_lock=bl)
                        print(f'  Batch failed on {b_used!r} (consecutive failures for this base: {nfail}): {exc!r}')
                    if not _bases_nonempty():
                        raise RuntimeError('All RunPod worker bases were removed after repeated batch failures (or none remain). Fix pods / LACUNA_DATA_API_BASE and resume.') from exc
                    busy_reroute = CLASSIFY_POD_BUSY_MARKER in str(exc)
                    if bl is not None:
                        with bl:
                            _nrot = len(bases)
                    else:
                        _nrot = len(bases)
                    excl = frozenset({_normalize_runpod_base_url(b_used)}) if busy_reroute and _nrot > 1 else frozenset()
                    b_new = _pick_base_for_submit(pending, ex, exclude_norm=excl)
                    print(f'\n[RunPod batch queue] re-queue {len(bi):,} items → {b_new} …')
                    fut2 = ex.submit(_run_http_batch_wrapped, b_new, bs, bi)
                    pending[fut2] = (bi, bs, b_new)

        def _pick_base_for_submit(pending: dict, ex: ThreadPoolExecutor, *, exclude_norm: frozenset[str] | None=None) -> str:
            nonlocal rr
            mip = _runpod_classify_max_in_flight_per_base()
            excl = exclude_norm or frozenset()
            while True:
                while len(pending) >= _effective_parallel_w():
                    _drain_one_future(pending, ex)
                if mip == 0:
                    return _pick_base_rr()
                if bl is not None:
                    with bl:
                        if not bases:
                            raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                        snap = list(bases)
                else:
                    if not bases:
                        raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                    snap = list(bases)
                if not snap:
                    raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                cnt = _count_inflight_by_norm(pending)
                n = len(snap)
                for offset in range(n):
                    i = (rr + offset) % n
                    b = snap[i]
                    kn = _normalize_runpod_base_url(b)
                    if kn in excl:
                        continue
                    if cnt.get(kn, 0) < mip:
                        rr = (i + 1) % n
                        return b
                _drain_one_future(pending, ex)

        def _flush_batch_queue(pending: dict, ex: ThreadPoolExecutor) -> None:
            if batch_queue is None:
                return
            while batch_queue:
                bi, bs = batch_queue[0]
                b_pick = _pick_base_for_submit(pending, ex)
                batch_queue.popleft()
                print(f'\n[RunPod batch queue] {len(bi):,} items → {b_pick} …')
                fut = ex.submit(_run_http_batch_wrapped, b_pick, bs, bi)
                pending[fut] = (bi, bs, b_pick)

        def _enqueue_stream_batch(pending: dict, ex: ThreadPoolExecutor, bi: list[int], bs: list[dict]) -> None:
            if batch_queue is None:
                b_pick = _pick_base_for_submit(pending, ex)
                print(f'\n[RunPod batch queue] {len(bi):,} items → {b_pick} …')
                fut = ex.submit(_run_http_batch_wrapped, b_pick, bs, bi)
                pending[fut] = (bi, bs, b_pick)
            else:
                while len(batch_queue) >= qmax_res:
                    _drain_one_future(pending, ex)
                    _flush_batch_queue(pending, ex)
                batch_queue.append((list(bi), list(bs)))
                _flush_batch_queue(pending, ex)
            while len(pending) >= _effective_parallel_w():
                _drain_one_future(pending, ex)
        with ThreadPoolExecutor(max_workers=parallel_w_cap) as ex:
            pending: dict = {}
            with open(input_path, 'r', encoding='utf-8') as rf:
                for i in range(total):
                    raw = rf.readline()
                    if not raw:
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    if i in already_done:
                        continue
                    batch_indices.append(i)
                    batch_samples.append(json.loads(line))
                    if len(batch_indices) >= batch_size:
                        bi = batch_indices
                        bs = batch_samples
                        batch_indices = []
                        batch_samples = []
                        _enqueue_stream_batch(pending, ex, bi, bs)
                if batch_indices:
                    if not _bases_nonempty():
                        raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                    bi = batch_indices
                    bs = batch_samples
                    _enqueue_stream_batch(pending, ex, bi, bs)
            while batch_queue and len(batch_queue) > 0:
                _flush_batch_queue(pending, ex)
                if batch_queue and len(pending) >= _effective_parallel_w():
                    _drain_one_future(pending, ex)
            while pending:
                _drain_one_future(pending, ex)
        if category_dir:
            print(f'  Done — per-category JSONL under {category_dir} (merged classified JSONL not written).')
        else:
            _write_output_from_disk(input_path, output_path, state['results'], total)
        elapsed = time.time() - start_time
        print(f'\n  {'█' * 30} 100.0% Complete\n  {total:,} samples in {elapsed / 60:.1f}min (RunPod HTTP, streamed)')
        return output_path
    finally:
        _runpod_clear_shared_bases_for_classify()

def batch_classify_runpod(input_path: str, output_path: str, *, base_url: str | None=None, api_key: str | None=None, state_path: str | None=None, batch_size: int=RUNPOD_CLASSIFY_BATCH_SIZE, max_samples: int | None=None, request_timeout: float=900.0, stream: bool | None=None, category_dir: str | None=None, clear_category_dir: bool=True, parallel_workers: int | None=None) -> str:
    """Classify via Lacuna data worker on RunPod (POST /classify/batch).

    Same ``.state.json`` + JSONL layout as ``batch_classify`` (non-RunPod batch). Do not
    mix backends on one ``output_path`` without clearing state: Anthropic
    ``pending_batch_ids`` are ignored here and cleared when resuming on RunPod.

    Set ``LACUNA_DATA_API_BASE`` in the repository root ``.env`` (e.g.
    ``https://<pod>-8000.proxy.runpod.net``). Optional ``LACUNA_DATA_API_KEY``
    is sent as ``X-API-Key`` when the worker enforces it.

    If ``max_samples`` is None, ``LACUNA_DATA_CLASSIFY_MAX_SAMPLES`` in the
    environment (if set) caps how many physical lines are classified.

    If ``stream`` is None, files >= 400 MiB are read in streaming mode (avoids
    loading the whole JSONL into RAM) — use for ``hf_datasets/.../train.jsonl``.

    If ``category_dir`` is set, each RunPod batch appends rows to
    ``identity.jsonl`` / ``values.jsonl`` / … there (original JSON only; no
    ``category`` field). The file at ``output_path`` is not created — use
    ``output_path`` only to derive ``.state.json`` (or pass ``state_path``).

    Multiple RunPod workers: set ``LACUNA_DATA_API_BASE`` to comma- or whitespace-separated
    proxy URLs, and optionally ``LACUNA_DATA_CLASSIFY_PARALLEL`` or ``parallel_workers``.
    Optional: ``LACUNA_DATA_CLASSIFY_MAX_IN_FLIGHT_PER_BASE`` (default 1; 0 = legacy unlimited
    stacking per URL), ``LACUNA_DATA_CLASSIFY_BUSY_REROUTE`` (default 1: HTTP 503 reroutes to
    another pod), ``LACUNA_DATA_CLASSIFY_PENDING_BATCH_QUEUE_MAX`` (extra batches buffered while
    waiting for a free pod; 0 = disabled).
    """
    batch_size = _runpod_classify_batch_size_resolved(batch_size)
    _reset_runpod_bad_base_tracking()
    max_samples = _classify_max_samples_from_env(max_samples)
    if max_samples is not None:
        print(f'  max_samples cap (env or arg): {max_samples:,}')
    if stream is None:
        try:
            stream = os.path.getsize(input_path) >= 400 * 1024 * 1024
        except OSError:
            stream = False
    if stream:
        return _batch_classify_runpod_stream(input_path=input_path, output_path=output_path, base_url=base_url, api_key=api_key, state_path=state_path, batch_size=batch_size, max_samples=max_samples, request_timeout=request_timeout, category_dir=category_dir, clear_category_dir=clear_category_dir, parallel_workers=parallel_workers)
    raw_bases = (base_url or os.environ.get('LACUNA_DATA_API_BASE', '')).strip()
    sb = _RUNPOD_CLASSIFY_SHARED_BASES
    bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
    if sb is not None:
        bases = sb
    else:
        bases = _parse_runpod_classifier_bases(raw_bases)
        if bases:
            _runpod_attach_shared_bases_for_classify(bases, threading.Lock())
        sb = _RUNPOD_CLASSIFY_SHARED_BASES
        bl = _RUNPOD_CLASSIFY_SHARED_BASES_LOCK
        if sb is not None:
            bases = sb
    if not bases:
        raise ValueError('Set LACUNA_DATA_API_BASE in .env (e.g. https://YOUR_POD-8000.proxy.runpod.net) or pass base_url=')
    try:
        if api_key is None:
            k = os.environ.get('LACUNA_DATA_API_KEY', '').strip()
            api_key = k or None
        if state_path is None:
            state_path = output_path + '.state.json'
        failure_log_path = _classify_batch_failure_log_path_resolved(state_path)
        with open(input_path, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        total = len(lines) if max_samples is None else min(len(lines), max_samples)
        lines = lines[:total]
        state, _state_corrupt_recover = _load_classify_state_file(state_path, context='RunPod HTTP classify')
        if state.get('results'):
            print(f'Resuming: {len(state['results']):,}/{total:,} already classified')
        state.setdefault('pending_batch_ids', [])
        if state['pending_batch_ids']:
            print('WARNING: Clearing pending_batch_ids from a prior non-RunPod batch run — not used with RunPod HTTP classification.')
            state['pending_batch_ids'] = []
            _atomic_json_dump_state(state_path, state)
        if category_dir and clear_category_dir and (not state.get('results')) and (not _state_corrupt_recover):
            _clear_category_jsonl(category_dir)
            print(f'  Cleared *.jsonl under {category_dir} (fresh run)')
        if len(state['results']) >= total:
            print(f'Already complete: {total:,}/{total:,}')
            if category_dir:
                _sync_category_dir_from_state(input_path, total, state['results'], category_dir)
                print(f'  Category files synced under {category_dir}')
            else:
                _write_output(lines, state['results'], output_path, total)
            return output_path
        target_n = max(1, int(os.environ.get('LACUNA_DATA_CLASSIFY_POD_COUNT', str(max(len(bases), 1)))))
        parallel_w_cap = _runpod_classify_parallel_workers(target_n, parallel_workers)

        def _effective_parallel_w() -> int:
            if bl is not None:
                with bl:
                    nb = len(bases)
            else:
                nb = len(bases)
            return min(parallel_w_cap, max(1, nb))

        def _bases_nonempty() -> bool:
            if bl is not None:
                with bl:
                    return bool(bases)
            return bool(bases)

        def _pick_base_rr() -> str:
            nonlocal rr
            if bl is not None:
                with bl:
                    if not bases:
                        raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                    b_pick = bases[rr % len(bases)]
                    rr += 1
            else:
                if not bases:
                    raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                b_pick = bases[rr % len(bases)]
                rr += 1
            return b_pick
        use_graphql_refresh = len(bases) == 1 and parallel_w_cap == 1
        base_holder = [bases[0]] if use_graphql_refresh else []
        print(f'RunPod classify: {len(bases):,} base URL(s) (more may join while async deploy finishes), up to {parallel_w_cap:,} concurrent batch(es) in flight' + (' (GraphQL URL refresh enabled)' if use_graphql_refresh else ''))
        _dbd = _runpod_bad_base_drop_threshold()
        if _dbd > 0:
            print(f'  Bad-base drop: after {_dbd} consecutive batch failure(s) on one URL, it is removed briefly; failed batches re-queue on others. Dropped-pod handling={_runpod_dropped_pod_handling()!r} (env LACUNA_DATA_CLASSIFY_DROPPED_POD_HANDLING=restart|stop|none). Set LACUNA_DATA_CLASSIFY_DROP_BAD_BASE_AFTER=0 to disable removal.')
        already_done = {int(k) for k in state['results'].keys()}
        remaining = [(i, json.loads(lines[i].strip())) for i in range(total) if i not in already_done and lines[i].strip()]
        print(f'Total: {total:,} | Already done: {len(already_done):,} | Remaining: {len(remaining):,}')
        chunks = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
        n_chunks = len(chunks)
        start_time = time.time()

        def _run_http_batch_ns(target_base: str, chunk: list[tuple[int, dict]]) -> list[dict]:
            samples = [s for _, s in chunk]
            line_indices = [i for i, _ in chunk]
            items = [{'text': _sample_to_preview(s)} for s in samples]
            extra = {'input_path': input_path, 'line_indices': line_indices, 'samples': samples}
            if use_graphql_refresh:
                return _post_classify_batch_http_resilient(base_holder, items, api_key, request_timeout, failure_log_path=failure_log_path, failure_extra=extra)
            return _post_classify_batch_http(target_base, items, api_key, request_timeout, failure_log_path=failure_log_path, failure_extra=extra)

        def _run_http_batch_ns_wrapped(target_base: str, chunk: list[tuple[int, dict]]) -> tuple[str, object]:
            try:
                return ('ok', _run_http_batch_ns(target_base, chunk))
            except BaseException as e:
                return ('err', e)

        def _apply_chunk_ns(chunk: list[tuple[int, dict]], http_results: list[dict], b_used: str, batch_no: int) -> None:
            if len(http_results) != len(chunk):
                raise RuntimeError(f'Expected {len(chunk)} results, got {len(http_results)}')
            labels_per_row: list[list[str]] = []
            for (idx, sample), row in zip(chunk, http_results):
                labs = _labels_from_classify_row(row, sample=sample)
                labels_per_row.append(labs)
            with state_lock:
                for (idx, _sample), labs in zip(chunk, labels_per_row):
                    state['results'][str(idx)] = labs
                state['completed_batches'].append(f'runpod:http:{batch_no}')
                _atomic_json_dump_state(state_path, state)
                print(f'  Checkpoint {len(state['results']):,}/{total:,} (batch #{batch_no}, via {b_used})')
                if category_dir:
                    samples = [sample for _, sample in chunk]
                    _append_batch_to_category_dir(category_dir, samples, labels_per_row)
                    print(f'  Appended batch to category files under {category_dir}')
        state_lock = threading.Lock()
        rr = 0
        batch_no = 0
        qmax_res_ns = _runpod_classify_pending_batch_queue_max_resolved(parallel_w_cap)
        batch_queue_ns: deque | None = deque() if qmax_res_ns > 0 else None

        def _count_inflight_by_norm_ns(pr: dict) -> dict[str, int]:
            out: dict[str, int] = {}
            for _fut, (_chunk, bu, _bn) in pr.items():
                k = _normalize_runpod_base_url(bu)
                out[k] = out.get(k, 0) + 1
            return out

        def _drain_one_future_ns(pending: dict, ex: ThreadPoolExecutor) -> None:
            done_set, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
            for fut in done_set:
                chunk, b_used, bn = pending.pop(fut)
                tag, payload = fut.result()
                if tag == 'ok':
                    _runpod_record_batch_success(b_used)
                    _apply_chunk_ns(chunk, payload, b_used, bn)
                else:
                    exc = payload
                    if CLASSIFY_FATAL_CUDA_MARKER in str(exc):
                        print(f'  Batch failed on {b_used!r} (fatal GPU/worker): {exc!r}')
                        _runpod_force_drop_fatal_cuda_worker(bases, b_used, api_key, bases_lock=bl)
                        nfail, _dropped = (1, True)
                    elif CLASSIFY_POD_BUSY_MARKER in str(exc):
                        print(f'  Pod busy (HTTP 503) on {b_used!r} — rerouting: {exc!r}')
                        nfail, _dropped = (0, False)
                    else:
                        nfail, _dropped = _runpod_record_batch_failure_and_maybe_drop(bases, b_used, worker_api_key=api_key, bases_lock=bl)
                        print(f'  Batch failed on {b_used!r} (consecutive failures for this base: {nfail}): {exc!r}')
                    if not _bases_nonempty():
                        raise RuntimeError('All RunPod worker bases were removed after repeated batch failures (or none remain). Fix pods / LACUNA_DATA_API_BASE and resume.') from exc
                    busy_reroute = CLASSIFY_POD_BUSY_MARKER in str(exc)
                    if bl is not None:
                        with bl:
                            _nrot = len(bases)
                    else:
                        _nrot = len(bases)
                    excl = frozenset({_normalize_runpod_base_url(b_used)}) if busy_reroute and _nrot > 1 else frozenset()
                    b_new = _pick_base_for_submit_ns(pending, ex, exclude_norm=excl)
                    print(f'\n[RunPod {bn}/{n_chunks}] re-queue {len(chunk):,} items → {b_new} …')
                    fut2 = ex.submit(_run_http_batch_ns_wrapped, b_new, chunk)
                    pending[fut2] = (chunk, b_new, bn)

        def _pick_base_for_submit_ns(pending: dict, ex: ThreadPoolExecutor, *, exclude_norm: frozenset[str] | None=None) -> str:
            nonlocal rr
            mip = _runpod_classify_max_in_flight_per_base()
            excl = exclude_norm or frozenset()
            while True:
                while len(pending) >= _effective_parallel_w():
                    _drain_one_future_ns(pending, ex)
                if mip == 0:
                    return _pick_base_rr()
                if bl is not None:
                    with bl:
                        if not bases:
                            raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                        snap = list(bases)
                else:
                    if not bases:
                        raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                    snap = list(bases)
                if not snap:
                    raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                cnt = _count_inflight_by_norm_ns(pending)
                n = len(snap)
                for offset in range(n):
                    i = (rr + offset) % n
                    b = snap[i]
                    kn = _normalize_runpod_base_url(b)
                    if kn in excl:
                        continue
                    if cnt.get(kn, 0) < mip:
                        rr = (i + 1) % n
                        return b
                _drain_one_future_ns(pending, ex)

        def _flush_batch_queue_ns(pending: dict, ex: ThreadPoolExecutor) -> None:
            if batch_queue_ns is None:
                return
            while batch_queue_ns:
                chunk, bno = batch_queue_ns[0]
                b_pick = _pick_base_for_submit_ns(pending, ex)
                batch_queue_ns.popleft()
                print(f'\n[RunPod {bno}/{n_chunks}] {len(chunk):,} items → {b_pick} …')
                fut = ex.submit(_run_http_batch_ns_wrapped, b_pick, chunk)
                pending[fut] = (chunk, b_pick, bno)

        def _enqueue_chunk_ns(pending: dict, ex: ThreadPoolExecutor, chunk: list[tuple[int, dict]], chunk_i: int) -> None:
            if batch_queue_ns is None:
                b_pick = _pick_base_for_submit_ns(pending, ex)
                print(f'\n[RunPod {chunk_i}/{n_chunks}] {len(chunk):,} items → {b_pick} …')
                fut = ex.submit(_run_http_batch_ns_wrapped, b_pick, chunk)
                pending[fut] = (chunk, b_pick, chunk_i)
            else:
                while len(batch_queue_ns) >= qmax_res_ns:
                    _drain_one_future_ns(pending, ex)
                    _flush_batch_queue_ns(pending, ex)
                batch_queue_ns.append((list(chunk), chunk_i))
                _flush_batch_queue_ns(pending, ex)
            while len(pending) >= _effective_parallel_w():
                _drain_one_future_ns(pending, ex)
        with ThreadPoolExecutor(max_workers=parallel_w_cap) as ex:
            pending: dict = {}
            for chunk_i, chunk in enumerate(chunks, start=1):
                if not _bases_nonempty():
                    raise RuntimeError('No RunPod bases left in rotation (all removed after failures).')
                _enqueue_chunk_ns(pending, ex, chunk, chunk_i)
                batch_no = chunk_i
            while batch_queue_ns and len(batch_queue_ns) > 0:
                _flush_batch_queue_ns(pending, ex)
                if batch_queue_ns and len(pending) >= _effective_parallel_w():
                    _drain_one_future_ns(pending, ex)
            while pending:
                _drain_one_future_ns(pending, ex)
        if category_dir:
            print(f'  Done — per-category JSONL under {category_dir} (merged classified JSONL not written).')
        else:
            _write_output(lines, state['results'], output_path, total)
        elapsed = time.time() - start_time
        print(f'\n  {'█' * 30} 100.0% Complete\n  {total:,} samples in {elapsed / 60:.1f}min (RunPod HTTP)')
        return output_path
    finally:
        _runpod_clear_shared_bases_for_classify()

def _write_output(lines: list[str], results: dict, output_path: str, total: int):
    """Write final classified JSONL from original lines + results."""
    with open(output_path, 'w', encoding='utf-8') as wf:
        for i in range(total):
            line = lines[i].strip()
            if not line:
                continue
            sample = json.loads(line)
            labs = _normalize_stored_labels(results.get(str(i)))
            sample['categories'] = labs
            sample['category'] = ','.join(labs)
            wf.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'  Output written: {output_path}')

def summarize_classification_from_category_dir(category_dir: str) -> None:
    """Print counts from ``identity.jsonl``, ``values.jsonl``, etc."""
    from collections import Counter
    counts: Counter = Counter()
    for cat in sorted(VALID_CATEGORIES):
        p = os.path.join(category_dir, f'{cat}.jsonl')
        if not os.path.isfile(p):
            counts[cat] = 0
            continue
        n = 0
        with open(p, 'r', encoding='utf-8') as f:
            for _ in f:
                n += 1
        counts[cat] = n
    total = sum(counts.values())
    print(f'Classification summary (under {category_dir}) — total lines: {total:,}')
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * n / total if total else 0.0
        print(f'  {cat:12s} {n:8,}  ({pct:5.1f}%)')
