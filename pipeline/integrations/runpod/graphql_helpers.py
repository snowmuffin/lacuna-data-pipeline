"""RunPod GraphQL helpers: list pods and resolve data worker proxy base URL."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# Cloudflare may return HTTP 403 / error 1010 for urllib's default User-Agent.
_GRAPHQL_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.runpod.io",
    "Referer": "https://www.runpod.io/",
}

# Treat as "may still reach HTTP ports" (booting or running).
_ACTIVE_POD_STATUSES = frozenset({"RUNNING", "CREATED", "RESTARTING"})

_STATUS_RANK = {"RUNNING": 0, "CREATED": 1, "RESTARTING": 2}


def _desired_status_active(desired: str) -> bool:
    """GraphQL sometimes varies casing; normalize before comparing."""
    s = (desired or "").strip().upper()
    return s in {x.upper() for x in _ACTIVE_POD_STATUSES}


def _status_rank(desired: str) -> int:
    s = (desired or "").strip().upper()
    for k, v in _STATUS_RANK.items():
        if k.upper() == s:
            return v
    return 9


def _ports_suggest_8000_exposed(ports: object, *, strict: bool) -> bool:
    """Whether we trust the pod can reach the worker on 8000 via the usual proxy URL.

    RunPod often omits or leaves ``ports`` empty while the pod is already reachable;
    if ``strict`` is False, missing/empty ``ports`` still counts as eligible.
    """
    if not strict:
        return True
    if ports is None:
        return True
    if isinstance(ports, str) and not ports.strip():
        return True
    if isinstance(ports, (list, dict)) and len(ports) == 0:
        return True
    blob = json.dumps(ports) if isinstance(ports, (list, dict)) else str(ports)
    return "8000" in blob

_LIST_PODS_QUERY = """
query ListDataWorkerPods {
  myself {
    pods {
      id
      name
      desiredStatus
      ports
      machine {
        podHostId
      }
      latestTelemetry {
        time
        averageGpuMetrics {
          percentUtilization
          memoryUtilization
        }
        individualGpuMetrics {
          percentUtilization
          memoryUtilization
        }
      }
    }
  }
}
"""


def pod_gpu_metrics_from_telemetry(pod: dict) -> dict | None:
    """GPU / VRAM utilization from RunPod ``latestTelemetry`` (no SSH).

    Returns a dict with keys ``gpu_util``, ``mem_pct``, ``mem_used``, ``mem_total``,
    ``error`` suitable for merging into GPU monitor rows. ``mem_*`` MiB are always
    ``None`` (API gives utilization % only). Returns ``None`` if telemetry is absent.
    """
    tel = pod.get("latestTelemetry")
    if not isinstance(tel, dict):
        return None

    def _from_avg(avg: dict) -> dict | None:
        pu = avg.get("percentUtilization")
        mu = avg.get("memoryUtilization")
        if pu is None and mu is None:
            return None
        gpu_u: int | None = None
        mem_pct: float | None = None
        if pu is not None:
            try:
                gpu_u = int(round(float(pu)))
            except (TypeError, ValueError):
                pass
        if mu is not None:
            try:
                mem_pct = round(float(mu), 1)
            except (TypeError, ValueError):
                pass
        return {
            "gpu_util": gpu_u,
            "mem_pct": mem_pct,
            "mem_used": None,
            "mem_total": None,
            "error": None,
        }

    avg = tel.get("averageGpuMetrics")
    if isinstance(avg, dict):
        out = _from_avg(avg)
        if out is not None:
            return out

    ind = tel.get("individualGpuMetrics")
    if not isinstance(ind, list) or not ind:
        return None
    gpus = [g for g in ind if isinstance(g, dict)]
    if not gpus:
        return None
    utils: list[float] = []
    mems: list[float] = []
    for g in gpus:
        pu = g.get("percentUtilization")
        if pu is not None:
            try:
                utils.append(float(pu))
            except (TypeError, ValueError):
                pass
        mu = g.get("memoryUtilization")
        if mu is not None:
            try:
                mems.append(float(mu))
            except (TypeError, ValueError):
                pass
    if not utils and not mems:
        return None
    gpu_u = int(round(sum(utils) / len(utils))) if utils else None
    mem_pct = round(sum(mems) / len(mems), 1) if mems else None
    return {
        "gpu_util": gpu_u,
        "mem_pct": mem_pct,
        "mem_used": None,
        "mem_total": None,
        "error": None,
    }


def pod_basic_ssh_username(pod: dict) -> str:
    """SSH username for RunPod basic proxy (``<user>@ssh.runpod.io``).

    RunPod's gateway expects ``machine.podHostId`` (e.g. ``abc12-6441103b``), which
    can differ from GraphQL ``id``. Falls back to ``id`` when ``podHostId`` is absent.
    """
    m = pod.get("machine")
    if isinstance(m, dict):
        ph = (m.get("podHostId") or "").strip()
        if ph:
            return ph
    return (pod.get("id") or "").strip()


def runpod_graphql(
    api_key: str, query: str, variables: dict | None = None
) -> dict:
    """POST a GraphQL query (and optional variables); same auth as deploy scripts."""
    key_q = urllib.parse.quote(api_key, safe="")
    url = f"{RUNPOD_GRAPHQL}?api_key={key_q}"
    payload: dict = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for hk, hv in _GRAPHQL_HEADERS.items():
        req.add_header(hk, hv)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()
        except Exception:
            detail = ""
        raise RuntimeError(f"RunPod GraphQL HTTP {e.code}: {detail}") from e


def list_my_pods(api_key: str) -> list[dict]:
    data = runpod_graphql(api_key, _LIST_PODS_QUERY)
    errs = data.get("errors")
    if errs:
        raise RuntimeError(errs)
    myself = data.get("data", {}).get("myself")
    if not myself:
        raise RuntimeError("RunPod GraphQL: missing data.myself")
    return myself.get("pods") or []


def discover_worker_base(
    api_key: str,
    *,
    pod_name: str | None = None,
    require_port_8000: bool = True,
) -> str | None:
    """Return ``https://{{pod_id}}-8000.proxy.runpod.net`` for a matching active pod, or ``None``.

    Matches pods where ``name`` equals ``pod_name`` (default: ``POD_NAME`` env or
    ``lacuna-data-worker``, same as ``runpod_deploy_worker.sh``),
    ``desiredStatus`` is RUNNING/CREATED/RESTARTING, and ``ports`` contains ``8000`` when
    ``require_port_8000`` is True.
    """
    name = (pod_name or os.environ.get("POD_NAME") or "lacuna-data-worker").strip()
    pods = list_my_pods(api_key)
    candidates: list[tuple[str, str]] = []
    for p in pods:
        if (p.get("name") or "").strip() != name:
            continue
        st = (p.get("desiredStatus") or "").strip()
        if not _desired_status_active(st):
            continue
        ports = p.get("ports")
        if not _ports_suggest_8000_exposed(ports, strict=require_port_8000):
            continue
        pid = (p.get("id") or "").strip()
        if not pid:
            continue
        candidates.append((pid, st))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (_status_rank(t[1]), t[0]))
    best_id = candidates[0][0]
    return f"https://{best_id}-8000.proxy.runpod.net"


def classify_worker_pod_name(stem: str, count: int, index: int) -> str:
    """Pod display name for classify slot ``index`` (1-based).

    Single worker: ``stem`` only (same as ``POD_NAME`` in deploy script).
    Multiple workers: ``stem-1``, ``stem-2``, …
    """
    if count <= 1:
        return stem
    return f"{stem}-{index}"


def graphql_pod_names_for_slot(stem: str, count: int, index: int) -> list[str]:
    """RunPod pod ``name`` values to try for this slot (highest priority first).

    When ``count > 1``, slot 1 tries ``stem-1`` then legacy ``stem`` (single pod from
    before multi-pod naming). Other slots use ``stem-{index}`` only.
    """
    stem = stem.strip()
    if count <= 1:
        return [stem]
    numbered = f"{stem}-{index}"
    if index == 1:
        return [numbered, stem]
    return [numbered]


def discover_classify_bases(
    api_key: str,
    stem: str,
    count: int,
    *,
    require_port_8000: bool = True,
) -> dict[int, str]:
    """Map slot index ``1..count`` → ``https://{{id}}-8000.proxy.runpod.net`` for active pods.

    Matches RunPod pod ``name`` via :func:`graphql_pod_names_for_slot` per slot.
    Missing slots are omitted from the dict.
    """
    if count < 1:
        return {}
    stem = stem.strip()
    pods = list_my_pods(api_key)
    by_name: dict[str, tuple[str, str]] = {}
    for p in pods:
        pname = (p.get("name") or "").strip()
        st = (p.get("desiredStatus") or "").strip()
        if not _desired_status_active(st):
            continue
        ports = p.get("ports")
        if not _ports_suggest_8000_exposed(ports, strict=require_port_8000):
            continue
        pid = (p.get("id") or "").strip()
        if not pname or not pid:
            continue
        rank = _status_rank(st)
        prev = by_name.get(pname)
        if prev is None or rank < _status_rank(prev[1]):
            by_name[pname] = (pid, st)

    out: dict[int, str] = {}
    for i in range(1, count + 1):
        for want in graphql_pod_names_for_slot(stem, count, i):
            t = by_name.get(want)
            if t:
                out[i] = f"https://{t[0]}-8000.proxy.runpod.net"
                break
    return out


def warn_if_classify_slots_missing(
    api_key: str,
    stem: str,
    count: int,
    found_slots: dict[int, str],
) -> None:
    """Log RunPod pod list when not all classify slots matched (name / status / ports hints)."""
    missing_idx = [i for i in range(1, count + 1) if i not in found_slots]
    if not missing_idx:
        return
    examples = [
        graphql_pod_names_for_slot(stem, count, i)[0] for i in missing_idx[:5]
    ]
    print(
        f"  Discovery matched {len(found_slots)}/{count} slot(s); "
        f"expected pod name(s) for missing slots include {examples!r} "
        f"(stem={stem!r}). Your RunPod pods (name | status | ports):"
    )
    try:
        pods = list_my_pods(api_key)
    except Exception as e:
        print(f"    (list_my_pods failed: {e!r})")
        return
    for p in pods[:80]:
        print(
            f"    {p.get('name')!r} | {p.get('desiredStatus')!r} | {p.get('ports')!r}"
        )


_PROXY_8000_RE = re.compile(
    r"^https://([a-zA-Z0-9]+)-8000\.proxy\.runpod\.net/?$", re.IGNORECASE
)

_POD_STOP_MUTATION = """
mutation PodStop($input: PodStopInput!) {
  podStop(input: $input) {
    id
    desiredStatus
  }
}
"""

_POD_RESUME_MUTATION = """
mutation PodResume($input: PodResumeInput!) {
  podResume(input: $input) {
    id
    desiredStatus
  }
}
"""

_POD_TERMINATE_MUTATION = """
mutation PodTerminate($input: PodTerminateInput!) {
  podTerminate(input: $input)
}
"""

# ``desiredStatus`` values that usually mean the pod is no longer running; safe to
# ``podTerminate`` to remove the pod record (see RunPod GraphQL ``PodStatus`` enum).
_TERMINATABLE_STATUSES_DEFAULT: frozenset[str] = frozenset(
    {"EXITED", "DEAD", "TERMINATED"}
)


def pod_id_from_worker_proxy_base(url: str) -> str | None:
    """Return pod id if ``url`` is ``https://{{id}}-8000.proxy.runpod.net``."""
    m = _PROXY_8000_RE.match((url or "").strip().rstrip("/"))
    return m.group(1) if m else None


def stop_pod(api_key: str, pod_id: str) -> dict:
    """GraphQL ``podStop`` for one pod id."""
    data = runpod_graphql(
        api_key,
        _POD_STOP_MUTATION,
        variables={
            "input": {"podId": pod_id, "incrementVersion": False},
        },
    )
    errs = data.get("errors")
    if errs:
        raise RuntimeError(errs)
    pod = data.get("data", {}).get("podStop")
    if not pod:
        raise RuntimeError(f"RunPod GraphQL podStop: unexpected response {data!r}")
    return pod


def resume_pod(api_key: str, pod_id: str) -> dict:
    """GraphQL ``podResume`` (e.g. after ``podStop`` on the same id)."""
    data = runpod_graphql(
        api_key,
        _POD_RESUME_MUTATION,
        variables={"input": {"podId": pod_id}},
    )
    errs = data.get("errors")
    if errs:
        raise RuntimeError(errs)
    pod = data.get("data", {}).get("podResume")
    if not pod:
        raise RuntimeError(f"RunPod GraphQL podResume: unexpected response {data!r}")
    return pod


def stop_then_resume_pod(api_key: str, pod_id: str, *, pause_sec: float = 3.0) -> None:
    """Stop a pod, wait briefly, then resume (same id; proxy URL usually unchanged)."""
    stop_pod(api_key, pod_id)
    time.sleep(max(0.0, pause_sec))
    resume_pod(api_key, pod_id)


def pod_display_name_for_id(api_key: str, pod_id: str) -> str | None:
    """Return RunPod pod ``name`` for ``pod_id``, or ``None`` if not listed."""
    want = (pod_id or "").strip()
    if not want:
        return None
    for p in list_my_pods(api_key):
        if (p.get("id") or "").strip() == want:
            n = (p.get("name") or "").strip()
            return n or None
    return None


def stop_then_terminate_pod(
    api_key: str,
    pod_id: str,
    *,
    pause_after_stop_sec: float = 2.0,
    poll_interval: float = 3.0,
    max_wait_sec: float = 180.0,
) -> None:
    """``podStop``, wait until the pod is no longer active, then ``podTerminate``.

    Use when the worker GPU is wedged (e.g. CUDA device-side assert): resume would
    keep a broken process; terminating frees the slot for a fresh deploy with the
    same pod name.
    """
    try:
        stop_pod(api_key, pod_id)
    except Exception:
        pass
    time.sleep(max(0.0, pause_after_stop_sec))
    deadline = time.time() + max(0.0, max_wait_sec)
    terminable = {s.upper() for s in _TERMINATABLE_STATUSES_DEFAULT}
    terminable.add("STOPPED")
    while time.time() < deadline:
        pods = list_my_pods(api_key)
        st_now: str | None = None
        for p in pods:
            if (p.get("id") or "").strip() == pod_id:
                st_now = (p.get("desiredStatus") or "").strip().upper()
                break
        if st_now is None:
            return
        inactive = st_now in terminable or not _desired_status_active(
            st_now or ""
        )
        if inactive:
            break
        time.sleep(max(0.5, poll_interval))
    terminate_pod(api_key, pod_id)


def terminate_pod(api_key: str, pod_id: str) -> None:
    """GraphQL ``podTerminate`` — removes a stopped / exited pod from your account."""
    data = runpod_graphql(
        api_key,
        _POD_TERMINATE_MUTATION,
        variables={"input": {"podId": pod_id}},
    )
    errs = data.get("errors")
    if errs:
        raise RuntimeError(errs)


def stop_all_active_pods(api_key: str, *, dry_run: bool = False) -> list[str]:
    """``podStop`` every pod whose ``desiredStatus`` looks active (RUNNING/CREATED/RESTARTING).

    Returns pod ids that were stopped (or would be stopped when ``dry_run`` is True).
    """
    pods = list_my_pods(api_key)
    stopped: list[str] = []
    for p in pods:
        pid = (p.get("id") or "").strip()
        if not pid:
            continue
        st = (p.get("desiredStatus") or "").strip()
        if not _desired_status_active(st):
            continue
        name = (p.get("name") or "").strip()
        if dry_run:
            print(f"[dry-run] would podStop {pid!r} name={name!r} status={st!r}")
            stopped.append(pid)
            continue
        try:
            stop_pod(api_key, pid)
            print(f"podStop ok {pid!r} name={name!r} (was {st!r})")
            stopped.append(pid)
        except Exception as e:
            print(f"WARNING: podStop failed {pid!r} name={name!r}: {e!r}")
    return stopped


def terminate_pods_with_statuses(
    api_key: str,
    statuses: frozenset[str],
    *,
    dry_run: bool = False,
) -> list[str]:
    """``podTerminate`` each pod whose ``desiredStatus`` (uppercased) is in ``statuses``.

    Typical use: remove stopped pods after ``podStop`` (often ``EXITED``). RunPod may
    still list pods in ``DEAD`` / ``TERMINATED`` until terminated.

    Returns pod ids terminated (or that would be terminated when ``dry_run`` is True).
    """
    pods = list_my_pods(api_key)
    want = {s.strip().upper() for s in statuses if s.strip()}
    done: list[str] = []
    for p in pods:
        pid = (p.get("id") or "").strip()
        if not pid:
            continue
        st_raw = (p.get("desiredStatus") or "").strip()
        st = st_raw.upper()
        if st not in want:
            continue
        name = (p.get("name") or "").strip()
        if dry_run:
            print(
                f"[dry-run] would podTerminate {pid!r} name={name!r} status={st_raw!r}"
            )
            done.append(pid)
            continue
        try:
            terminate_pod(api_key, pid)
            print(f"podTerminate ok {pid!r} name={name!r} (was {st_raw!r})")
            done.append(pid)
        except Exception as e:
            print(f"WARNING: podTerminate failed {pid!r} name={name!r}: {e!r}")
    return done


def terminate_stopped_pods(api_key: str, *, dry_run: bool = False) -> list[str]:
    """Terminate pods in default non-running statuses (``EXITED``, ``DEAD``, ``TERMINATED``)."""
    return terminate_pods_with_statuses(
        api_key, _TERMINATABLE_STATUSES_DEFAULT, dry_run=dry_run
    )


def stop_classify_pods_from_proxy_bases(
    api_key: str, base_urls: list[str]
) -> list[str]:
    """Stop every unique pod referenced by worker proxy base URLs.

    Returns pod ids that were passed to ``podStop`` (duplicates in ``base_urls`` are skipped).
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for u in base_urls:
        pid = pod_id_from_worker_proxy_base(u)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
    for pid in ordered:
        stop_pod(api_key, pid)
    return ordered
