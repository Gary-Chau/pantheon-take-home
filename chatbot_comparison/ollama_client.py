"""Minimal client for the local Ollama HTTP API (https://github.com/ollama/ollama).

Ollama must be running locally (`ollama serve`, or the desktop app) before
using this client. No API key is required since everything runs on-device.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import requests


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 900):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_alive(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any],
        think: bool = False,
    ) -> Dict[str, Any]:
        """Calls POST /api/chat (non-streaming) and returns the raw JSON
        response with an added wall-clock timing measurement."""
        payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "think": think,
            "stream": False,
        }
        start = time.time()
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        wall_clock_seconds = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        data["_wall_clock_seconds"] = wall_clock_seconds
        return data
