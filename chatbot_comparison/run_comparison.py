"""Runs every task in configs/tasks/default.yaml against every model in
configs/models/default.yaml via the local Ollama API, and saves each
individual response as its own JSON file under result/raw/ for later
aggregation (see aggregate_to_excel.py).

Usage:
    python run_comparison.py                  # run everything
    python run_comparison.py dry_run=true      # preview planned calls, call nothing
    python run_comparison.py generation.temperature=0.2   # override any config value
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import hydra
from omegaconf import DictConfig, OmegaConf

from ollama_client import OllamaClient


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)


def find_existing_result(raw_result_dir: str, task_id: str, model: dict) -> tuple:
    """Returns (path, record) for an existing result, including old model tags."""
    default_name = f"{task_id}__{safe_filename(model['tag'])}.json"
    default_path = os.path.join(raw_result_dir, default_name)
    candidates = [default_path]
    candidates.extend(
        os.path.join(raw_result_dir, name)
        for name in os.listdir(raw_result_dir)
        if name.startswith(f"{task_id}__") and name.endswith(".json")
    )

    for path in dict.fromkeys(candidates):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            if record.get("model_family") == model["family"]:
                return path, record
        except (OSError, json.JSONDecodeError):
            continue

    return default_path, None


def run_single(
    client: OllamaClient,
    task: dict,
    model: dict,
    generation_options: dict,
    think: bool,
) -> dict:
    print(f"  -> [{model['display_name']}] {task['id']} ...", end="", flush=True)
    try:
        raw = client.chat(
            model["tag"], task["messages"], generation_options, think=think
        )
        response_text = raw.get("message", {}).get("content", "")
        record = {
            "task_id": task["id"],
            "category": task["category"],
            "technique_notes": task.get("technique_notes", ""),
            "model_tag": model["tag"],
            "model_family": model["family"],
            "model_display_name": model["display_name"],
            "messages": task["messages"],
            "generation_options": generation_options,
            "think": think,
            "response": response_text,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_metadata": {
                "total_duration_ns": raw.get("total_duration"),
                "load_duration_ns": raw.get("load_duration"),
                "prompt_eval_count": raw.get("prompt_eval_count"),
                "prompt_eval_duration_ns": raw.get("prompt_eval_duration"),
                "eval_count": raw.get("eval_count"),
                "eval_duration_ns": raw.get("eval_duration"),
                "wall_clock_seconds": raw.get("_wall_clock_seconds"),
            },
            "error": None,
        }
        print(" done")
    except Exception as exc:  # noqa: BLE001 - keep the batch going on any single failure
        record = {
            "task_id": task["id"],
            "category": task["category"],
            "technique_notes": task.get("technique_notes", ""),
            "model_tag": model["tag"],
            "model_family": model["family"],
            "model_display_name": model["display_name"],
            "messages": task["messages"],
            "generation_options": generation_options,
            "think": think,
            "response": None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_metadata": None,
            "error": str(exc),
        }
        print(f" FAILED ({exc})")
    return record


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    tasks = OmegaConf.to_container(cfg.tasks, resolve=True)
    models = OmegaConf.to_container(cfg.models, resolve=True)
    generation_options = OmegaConf.to_container(cfg.generation, resolve=True)

    os.makedirs(cfg.raw_result_dir, exist_ok=True)
    client = OllamaClient(host=cfg.ollama.host, timeout=cfg.ollama.timeout_seconds)

    if not cfg.dry_run and not client.is_alive():
        raise SystemExit(
            f"Could not reach Ollama at {cfg.ollama.host}. "
            "Start it with `ollama serve` (or the Ollama desktop app) and try again."
        )

    total = len(tasks) * len(models)
    print(f"Planned calls: {len(tasks)} tasks x {len(models)} models = {total}\n")

    for task in tasks:
        print(f"Task: {task['id']} ({task['category']})")
        for model in models:
            out_path, existing = find_existing_result(
                cfg.raw_result_dir, task["id"], model
            )

            if cfg.dry_run:
                print(f"  -> [DRY RUN] would call {model['tag']} for task '{task['id']}'")
                continue

            if (
                cfg.skip_successful
                and existing
                and not existing.get("error")
                and existing.get("response")
                and (
                    model["family"] != "Qwen"
                    or existing.get("think") == bool(cfg.ollama.think)
                )
            ):
                print(f"  -> [{model['display_name']}] {task['id']} ... skipped (already successful)")
                continue

            record = run_single(
                client, task, model, generation_options, think=cfg.ollama.think
            )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    if not cfg.dry_run:
        print(f"\nAll responses saved under {cfg.raw_result_dir}/")
        print("Next: python aggregate_to_excel.py")


if __name__ == "__main__":
    main()
