"""JSONL shaping helpers for the refine stage (multi/single-turn sync, split, dedup).

Upstream logic was developed in an external ``filter`` notebook (cells 4–7, 9, 11);
RunPod / upload steps are omitted. Pass directory paths as function arguments from the refine flow.
"""



# ----- notebook cell 4 -----

from typing import Any
import os
import json
import shutil
import tempfile
import re

def _preview_jsonl_line(line: str, max_chars: int=220) -> str:
    """Truncate a JSONL row for logging (avoid multi-MB notebook output)."""
    s = line.strip().replace('\n', ' ')
    if len(s) <= max_chars:
        return s
    return f'{s[:max_chars]}... (+{len(s) - max_chars} chars)'

def r_count_keys(data: dict, count: dict):
    for key, value in data.items():
        if key not in count:
            count[key] = 0
        count[key] += 1
        if isinstance(value, dict):
            count.update(r_count_keys(value, count))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    count.update(r_count_keys(item, count))
    return count

def _looks_like_chat_row(data: dict) -> bool:
    """True if the object plausibly holds sharegpt/OpenAI-style chat (not DPO / plain instruct)."""
    if isinstance(data.get('messages'), list) and len(data['messages']) > 0:
        return True
    if isinstance(data.get('conversation'), list) and len(data['conversation']) > 0:
        return True
    if isinstance(data.get('conversations'), list) and len(data['conversations']) > 0:
        return True
    if data.get('conversation_a') is not None and data.get('conversation_b') is not None:
        return True
    tex = data.get('text')
    if isinstance(tex, str) and tex.strip():
        return True
    return False

def check_multi_turn(line: str) -> bool:
    """Heuristic: repeated keys anywhere in the JSON tree (loose signal)."""
    keys: dict[str, int] = {}
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            keys = r_count_keys(data, keys)
            for _key, value in keys.items():
                if value > 1:
                    return True
            return False
    except Exception as e:
        print(f'exception: {e} {_preview_jsonl_line(line)}')
        return False
    return False

def is_single_key_data(line: str):
    pass_list = ['id', 'source']
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            for pass_key in pass_list:
                data.pop(pass_key, None)
            if len(data) == 1:
                return True
            else:
                return False
        return False
    except Exception as e:
        print(f'exception: {e}')
        return False

def check_multi_turn_file(file_path: str):
    try:
        last_nonempty: str | None = None
        with open(file_path, 'r', encoding='utf-8') as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                last_nonempty = line
                if is_single_key_data(line):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f'not multi-turn (invalid JSON): {file_path}')
                    return False
                if not isinstance(row, dict) or not _looks_like_chat_row(row):
                    print(f'not multi-turn: {file_path}')
                    return False
                if not check_multi_turn(line):
                    print(f'not multi-turn: {file_path}')
                    return False
        if last_nonempty is None:
            return False
        print(f'multi-turn: {file_path}')
        return True
    except Exception as e:
        print(f'exception: {e} {file_path}')
        return False

def check_multi_turn_dir(dir_path: str):
    multi_turn_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if check_multi_turn_file(file_path):
                multi_turn_files.append(file_path)
    print(f'total multi turn files: {len(multi_turn_files)}')
    return multi_turn_files
ROLE_TAGS = {'sys': 'system', 'usr': 'user', 'bot': 'assistant'}
TAG_PATTERN = re.compile('<\\s*(sys|usr|bot)\\s*>', re.IGNORECASE)

def parse_tagged_text_to_messages(text: str) -> list[dict[str, str]]:
    """Parse tagged text like "<sys>...
<usr>...
<bot>..." into messages."""
    messages: list[dict[str, str]] = []
    current_role: str | None = None
    pos = 0
    for match in TAG_PATTERN.finditer(text):
        tag_raw = match.group(1)
        tag = tag_raw.lower()
        start = match.start()
        if current_role is not None:
            chunk = text[pos:start]
            content = chunk.strip()
            if content:
                messages.append({'role': current_role, 'content': content})
        current_role = ROLE_TAGS[tag]
        pos = match.end()
    if current_role is not None:
        chunk = text[pos:]
        content = chunk.strip()
        if content:
            messages.append({'role': current_role, 'content': content})
    return messages
_SFT_STRIP_TOP_LEVEL_KEYS = frozenset({'openai_moderation', 'detoxify_moderation', 'conversation_id', 'model', 'turn', 'language', 'redacted', 'timestamp', 'toxic', 'model_a', 'model_b', 'winner', 'question_id', 'conversation_a', 'conversation_b', 'conversation', 'conversations', 'text', 'user_id', 'anonymized_user_id', 'uid', 'tstamp', 'judge', 'anony', 'toxic_chat_tag'})

def _coerce_message_list(value: Any) -> list[dict[str, Any]] | None:
    """Parse HF-style conversation (list or JSON string) into message dicts."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, list) or not value:
        return None
    if not all((isinstance(x, dict) for x in value)):
        return None
    return value

def _arena_winner_messages_for_sft(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Pairwise Arena -> chosen side only for SFT. Skips ties and unknown winners."""
    ca = _coerce_message_list(data.get('conversation_a'))
    cb = _coerce_message_list(data.get('conversation_b'))
    if ca is None or cb is None:
        return None
    w = data.get('winner')
    if w is None:
        return None
    if isinstance(w, str):
        wl = w.strip().lower()
        if wl in ('tie', 'draw') or wl.startswith('tie'):
            return None
    ma, mb = (data.get('model_a'), data.get('model_b'))
    pick: list[dict[str, Any]] | None = None
    if isinstance(w, int):
        if w == 0:
            pick = ca
        elif w == 1:
            pick = cb
    elif isinstance(w, str):
        wl = w.strip().lower()
        if wl in ('model_a', 'a', 'left', 'first'):
            pick = ca
        elif wl in ('model_b', 'b', 'right', 'second'):
            pick = cb
        elif ma is not None and w == ma:
            pick = ca
        elif mb is not None and w == mb:
            pick = cb
    elif w == ma:
        pick = ca
    elif w == mb:
        pick = cb
    return pick

def _strip_sft_metadata(data: dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    for k in list(data.keys()):
        if k != 'messages' and k in _SFT_STRIP_TOP_LEVEL_KEYS:
            data.pop(k, None)

def transform_line(line: str, removal_list: list[str], sync_target_keys: list[str], sync_target: str, *, sft_strip_metadata: bool=True) -> dict[str, Any] | None:
    """Transform one JSONL line into unified multi-turn format for SFT.

    - Chatbot Arena-style rows (``conversation_a`` / ``conversation_b`` / ``winner``):
      keep the human-preferred branch only; ties are skipped.
    - Otherwise: hoist the first matching key in ``sync_target_keys`` into
      ``sync_target`` (e.g. ``conversation`` / ``conversations`` / ``text`` -> ``messages``).
    - Optionally strip large metadata keys (moderation blobs, ids) so each line is
      mostly ``messages`` for fine-tuning.
    """
    line = line.strip()
    if not line:
        return None
    data = json.loads(line)
    for key in removal_list:
        data.pop(key, None)
    existing = data.get(sync_target)
    if isinstance(existing, list) and len(existing) > 0 and all((isinstance(m, dict) for m in existing)):
        _strip_sft_metadata(data, sft_strip_metadata)
        return data
    chosen = _arena_winner_messages_for_sft(data)
    if chosen is not None:
        out = {sync_target: chosen}
        _strip_sft_metadata(out, sft_strip_metadata)
        return out
    for key in sync_target_keys:
        if key not in data:
            continue
        value = data.pop(key)
        if isinstance(value, list) and all((isinstance(item, dict) for item in value)):
            data[sync_target] = value
            break
        if isinstance(value, str):
            msgs = parse_tagged_text_to_messages(value)
            if msgs:
                data[sync_target] = msgs
                break
            return None
        return None
    else:
        return None
    _strip_sft_metadata(data, sft_strip_metadata)
    return data



def check_single_turn_dir(dir_path: str):
    single_turn_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if check_single_turn_file(file_path):
                single_turn_files.append(file_path)

    print(f"total single turn files: {len(single_turn_files)}")
    return single_turn_files


def check_single_turn_file(file_path: str):
    try:
        last_nonempty: str | None = None
        with open(file_path, "r", encoding="utf-8") as fp:
            for raw in fp:
                line = raw.strip()
                if not line:
                    continue
                last_nonempty = line
                if not json.loads(line):
                    print(f"wrong json: {file_path}")
                    return True
                if not is_single_key_data(line) and not check_multi_turn(line):
                    print(f"not multi-turn: {file_path}")
                    return True
        if last_nonempty is None:
            return False
        print(f"multi-turn: {file_path}")
        return False

    except Exception as e:
        print(f"exception: {e} {file_path}")
        return False


def sync_multi_turn_files_for_list(files: list[str], base_dir: str, target_dir: str, sync_target_keys: list[str], removal_list: list[str], sync_target: str, *, sft_strip_metadata: bool=True) -> None:
    base_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(target_dir)
    use_temp = base_abs == target_abs
    temp_root: str | None = None
    actual_target_dir = target_dir
    if use_temp:
        temp_root = tempfile.mkdtemp(prefix='sync_multi_turn_tmp_')
        actual_target_dir = temp_root
    for file in files:
        file_abs = os.path.abspath(file)
        if not file_abs.startswith(base_abs + os.sep):
            print(f'Skip file outside base_dir: {file}')
            continue
        rel_path = os.path.relpath(file_abs, base_abs)
        output_path = os.path.join(actual_target_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        n_out = 0
        with open(file_abs, 'r', encoding='utf-8') as rf, open(output_path, 'w', encoding='utf-8') as wf:
            for line in rf:
                try:
                    transformed = transform_line(line=line, removal_list=removal_list, sync_target_keys=sync_target_keys, sync_target=sync_target, sft_strip_metadata=sft_strip_metadata)
                    if transformed is None:
                        continue
                    wf.write(json.dumps(transformed, ensure_ascii=False) + '\n')
                    n_out += 1
                except Exception as e:
                    print(f'exception: {e}')
                    break
        if n_out:
            print(f'synced {file_abs} -> {output_path} ({n_out:,} lines)')
    if use_temp and temp_root is not None:
        for root, dirs, files in os.walk(temp_root):
            for file in files:
                tmp_path = os.path.join(root, file)
                rel = os.path.relpath(tmp_path, temp_root)
                final_path = os.path.join(target_abs, rel)
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(tmp_path, final_path)
        shutil.rmtree(temp_root, ignore_errors=True)

def sync_message_format(dir_path: str, target_dir: str, target_root_key: str, target_sub_keys: list[str], removal_list: list[str], sync_target: str) -> None:
    """Normalize message format in multi-turn JSONL files.

    For each file under dir_path that is detected as multi-turn:
    - Look for target_root_key (e.g. "messages") which should be a list.
    - For each dict item in that list:
      * Remove keys listed in removal_list.
      * Find the first key from target_sub_keys that exists and move its value
        into sync_target.
      * Only keep items where the chosen value is valid (e.g. non-empty string).
    - Write the normalized JSONL to target_dir, preserving the relative path.
    """
    files = check_multi_turn_dir(dir_path)
    base_abs = os.path.abspath(dir_path)
    target_abs = os.path.abspath(target_dir)
    use_temp = base_abs == target_abs
    temp_root: str | None = None
    actual_target_dir = target_dir
    if use_temp:
        temp_root = tempfile.mkdtemp(prefix='sync_message_format_tmp_')
        actual_target_dir = temp_root
    for file in files:
        file_abs = os.path.abspath(file)
        if not file_abs.startswith(base_abs + os.sep):
            print(f'Skip file outside base_dir: {file_abs}')
            continue
        rel_path = os.path.relpath(file_abs, base_abs)
        out_path = os.path.join(actual_target_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        n_out = 0
        with open(file_abs, 'r', encoding='utf-8') as rf, open(out_path, 'w', encoding='utf-8') as wf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    data: dict[str, Any] = json.loads(line)
                except Exception as e:
                    print(f'Failed to parse JSON in {file_abs}: {e}')
                    continue
                if target_root_key not in data or not isinstance(data[target_root_key], list):
                    wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                    n_out += 1
                    continue
                items = data[target_root_key]
                new_items: list[dict[str, Any]] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    msg = dict(item)
                    for k in removal_list:
                        msg.pop(k, None)
                    chosen_value: Any | None = None
                    for k in target_sub_keys:
                        if k in msg:
                            chosen_value = msg.pop(k)
                            break
                    if chosen_value is None:
                        continue
                    if isinstance(chosen_value, str) and (not chosen_value.strip()):
                        continue
                    msg[sync_target] = chosen_value
                    new_items.append(msg)
                if new_items:
                    data[target_root_key] = new_items
                wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                n_out += 1
        print(f'normalized message format: {file_abs} -> {out_path} ({n_out:,} lines)')
    if use_temp and temp_root is not None:
        for root, dirs, files in os.walk(temp_root):
            for file in files:
                tmp_path = os.path.join(root, file)
                rel = os.path.relpath(tmp_path, temp_root)
                final_path = os.path.join(target_abs, rel)
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(tmp_path, final_path)
        shutil.rmtree(temp_root, ignore_errors=True)

def sync_multi_turn_files(dir_path: str, target_dir: str, sync_target_keys: list[str], removal_list: list[str], sync_target: str, *, sft_strip_metadata: bool=True) -> None:
    files = check_multi_turn_dir(dir_path)
    sync_multi_turn_files_for_list(files=files, base_dir=dir_path, target_dir=target_dir, sync_target_keys=sync_target_keys, removal_list=removal_list, sync_target=sync_target, sft_strip_metadata=sft_strip_metadata)


# ----- notebook cell 5 -----

from typing import Any
import json
import os
import tempfile
import shutil


def normalize_single_turn_record(
    data: dict[str, Any],
    instruction_keys: list[str],
    output_keys: list[str],
    input_keys: list[str],
    removal_list: list[str],
) -> dict[str, Any] | None:
    """Normalize various single-turn schemas into {instruction, input, output}.

    - instruction_keys: priority list of keys treated as the user instruction/question.
    - output_keys: priority list of keys treated as the assistant output/answer.
    - input_keys: keys that may contain optional extra input (defaults to "").
    - removal_list: keys to drop entirely (e.g. id, url, source).
    """
    obj = dict(data)

    for k in removal_list:
        obj.pop(k, None)

    instr = None
    for k in instruction_keys:
        if k in obj:
            instr = obj.pop(k)
            break

    out = None
    for k in output_keys:
        if k in obj:
            out = obj.pop(k)
            break

    inp: str = ""
    for k in input_keys:
        if k in obj:
            inp = obj.pop(k)
            break

    if not isinstance(instr, str) or not instr.strip():
        return None
    if not isinstance(out, str) or not out.strip():
        return None

    return {
        "instruction": instr,
        "input": inp,
        "output": out,
    }


def sync_single_turn_files(
    dir_path: str,
    target_dir: str,
    instruction_keys: list[str] | None = None,
    output_keys: list[str] | None = None,
    input_keys: list[str] | None = None,
    removal_list: list[str] | None = None,
) -> None:
    """Normalize *single-turn* JSONL files under dir_path into a unified schema.

    The unified schema per line is: {"instruction", "input", "output"}.
    If dir_path == target_dir, writes via a temp directory and then replaces
    originals to avoid partial-overwrite issues.
    """
    if instruction_keys is None:
        instruction_keys = ["instruction", "question"]
    if output_keys is None:
        output_keys = ["output", "answer", "solution"]
    if input_keys is None:
        input_keys = ["input"]
    if removal_list is None:
        removal_list = ["id", "url", "source"]

    base_abs = os.path.abspath(dir_path)
    target_abs = os.path.abspath(target_dir)

    use_temp = base_abs == target_abs
    temp_root: str | None = None
    actual_target_dir = target_dir

    if use_temp:
        temp_root = tempfile.mkdtemp(prefix="sync_single_turn_tmp_")
        actual_target_dir = temp_root

    files = check_single_turn_dir(dir_path)

    for file_path in files:
        file_abs = os.path.abspath(file_path)
        if not file_abs.startswith(base_abs + os.sep):
            print(f"Skip file outside base_dir: {file_abs}")
            continue

        rel_path = os.path.relpath(file_abs, base_abs)
        out_path = os.path.join(actual_target_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        n_out = 0
        with open(file_abs, "r", encoding="utf-8") as rf, open(
            out_path, "w", encoding="utf-8"
        ) as wf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Failed to parse JSON in {file_abs}: {e}")
                    continue
                if not isinstance(data, dict):
                    continue
                norm = normalize_single_turn_record(
                    data=data,
                    instruction_keys=instruction_keys,
                    output_keys=output_keys,
                    input_keys=input_keys,
                    removal_list=removal_list,
                )
                if norm is None:
                    continue
                wf.write(json.dumps(norm, ensure_ascii=False) + "\n")
                n_out += 1

        if n_out:
            print(f"normalized single-turn: {file_abs} -> {out_path} ({n_out:,} lines)")

    if use_temp and temp_root is not None:
        for root, dirs, files in os.walk(temp_root):
            for file in files:
                tmp_path = os.path.join(root, file)
                rel = os.path.relpath(tmp_path, temp_root)
                final_path = os.path.join(target_abs, rel)
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.move(tmp_path, final_path)
        shutil.rmtree(temp_root, ignore_errors=True)


# ----- notebook cell 6 -----

import os
import shutil
import tempfile
from pathlib import Path

from convmerge.convert import convert_with_config
from convmerge.preset import load_convert_preset

from pipeline.sources_config import REPO_ROOT

_PRESET_ALPACA_TO_MESSAGES = load_convert_preset(
    REPO_ROOT / "config" / "presets" / "alpaca_to_messages.yaml"
)


def sync_single_turn_to_multi_turn(single_turn_dir: str, target_dir: str) -> None:
    """Alpaca-style JSONL -> ``messages`` JSONL via ``presets/alpaca_to_messages.yaml``."""
    base_abs = os.path.abspath(single_turn_dir)
    target_abs = os.path.abspath(target_dir)
    use_temp = base_abs == target_abs
    temp_root: str | None = None
    actual_target_dir = target_dir
    if use_temp:
        temp_root = tempfile.mkdtemp(prefix="single_to_multi_tmp_")
        actual_target_dir = temp_root

    for root, _dirs, files in os.walk(base_abs):
        for file in files:
            file_abs = os.path.join(root, file)
            if not file_abs.endswith(".jsonl"):
                continue
            if not file_abs.startswith(base_abs + os.sep):
                print(f"Skip file outside base_dir: {file_abs}")
                continue
            rel_path = os.path.relpath(file_abs, base_abs)
            out_path = os.path.join(actual_target_dir, rel_path)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            try:
                _n_in, n_out = convert_with_config(
                    Path(file_abs),
                    Path(out_path),
                    _PRESET_ALPACA_TO_MESSAGES,
                )
            except Exception as e:
                print(f"skip {file_abs}: {e}")
                continue
            if n_out:
                print(f"single->multi: {file_abs} -> {out_path} ({n_out:,} lines)")

    if use_temp and temp_root is not None:
        for root, _dirs, files in os.walk(temp_root):
            for file in files:
                tmp_path = os.path.join(root, file)
                rel = os.path.relpath(tmp_path, temp_root)
                final_path = os.path.join(target_abs, rel)
                os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
                shutil.move(tmp_path, final_path)
        shutil.rmtree(temp_root, ignore_errors=True)


# ----- notebook cell 7 -----

import json
import os

from convmerge.normalize.turns import count_turns


def filter_empty_turns(input_path: str, output_path: str, min_turns: int = 1) -> None:
    """Remove samples with fewer than ``min_turns`` assistant messages."""
    total = 0
    kept = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as rf, \
            open(output_path, "w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            sample = json.loads(line)
            if count_turns(sample) >= min_turns:
                wf.write(line + "\n")
                kept += 1
    removed = total - kept
    pct = (removed / total * 100) if total else 0
    print(f"Total: {total:,} | Kept: {kept:,} | Removed: {removed:,} ({pct:.2f}%)")


# ----- notebook cell 9 -----

import json

import numpy as np


def collect_all_samples(
    source_dirs: list[str],
    out_path: str,
) -> str:
    """Collect all valid JSONL lines from source_dirs into a single file.

    Returns the output file path.
    """
    all_files: list[str] = []
    for d in source_dirs:
        d_abs = os.path.abspath(d)
        if not os.path.isdir(d_abs):
            print(f"skip missing dir: {d_abs}")
            continue
        for root, dirs, files in os.walk(d_abs):
            for f in files:
                if not f.endswith(".jsonl"):
                    continue
                all_files.append(os.path.join(root, f))

    print(f"found {len(all_files)} jsonl files")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for path in all_files:
            with open(path, "r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except Exception:
                        skipped += 1
                        continue
                    wf.write(line + "\n")
                    total += 1

    print(f"collected {total:,} samples -> {out_path}  (skipped {skipped:,} broken lines)")
    return out_path


def split_dataset(
    input_path: str,
    out_dir: str,
    train_ratio: float = 0.98,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[str, str]:
    """Shuffle-split JSONL into train/test with bounded RAM (numpy masks).

    Two passes over the file: count non-empty lines, then stream lines while
    applying a random train/test assignment. ``max_samples`` subsamples line
    indices before the split (same idea as shuffle + truncate).

    Returns (train_path, test_path).
    """
    with open(input_path, "r", encoding="utf-8") as rf:
        n_lines = sum(1 for line in rf if line.strip())
    if n_lines == 0:
        raise RuntimeError(f"no samples found in {input_path}")

    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples < n_lines:
        m = max_samples
        chosen = np.zeros(n_lines, dtype=np.bool_)
        chosen[rng.choice(n_lines, size=m, replace=False)] = True
    else:
        chosen = np.ones(n_lines, dtype=np.bool_)
        m = n_lines

    n_train = int(m * train_ratio)
    sub_pos = np.flatnonzero(chosen)
    rng.shuffle(sub_pos)
    in_train = np.zeros(n_lines, dtype=np.bool_)
    in_train[sub_pos[:n_train]] = True

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")

    ti = 0
    with open(input_path, "r", encoding="utf-8") as rf, open(
        train_path, "w", encoding="utf-8"
    ) as wtr, open(test_path, "w", encoding="utf-8") as wte:
        for line in rf:
            if not line.strip():
                continue
            if not chosen[ti]:
                ti += 1
                continue
            out = wtr if in_train[ti] else wte
            out.write(line if line.endswith("\n") else line + "\n")
            ti += 1

    print(f"split {m:,} samples (ratio {train_ratio})")
    print(f"  train: {n_train:,} -> {train_path}")
    print(f"  test:  {m - n_train:,} -> {test_path}")

    return train_path, test_path


# ----- notebook cell 11 -----

import os

from convmerge.normalize.dedup import deduplicate_jsonl


def duplication_filter(input_file_path: str, output_file_path: str) -> None:
    """Drop rows whose full JSON payload hashes to an already-seen MD5.

    Delegates to ``convmerge.normalize.dedup.deduplicate_jsonl``; the hashing
    contract (md5 over ``json.dumps(data, sort_keys=True)``) matches the
    previous implementation.
    """
    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)
    total, kept = deduplicate_jsonl(input_file_path, output_file_path, algorithm="md5")
    removed = total - kept
    pct = (removed / total * 100) if total else 0
    print(f"Total: {total:,} | Kept: {kept:,} | Removed: {removed:,} ({pct:.2f}%)")
