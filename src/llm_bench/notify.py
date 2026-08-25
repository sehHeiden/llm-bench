"""
Push a message to ntfy. Topic/server come from config/ntfy.env (pydantic-settings).

Usage: uv run notify "message text"
"""

import sys
import urllib.request
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ntfy server + topic, read from config/ntfy.env."""

    ntfy_url: str = "https://ntfy.sh"
    ntfy_topic: str = ""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / "config" / "ntfy.env"),
        extra="ignore",
    )


def notify(msg: str) -> None:
    """Push a message to ntfy (topic from config/ntfy.env)."""
    settings = Settings()
    if not settings.ntfy_topic:
        print("ntfy_topic fehlt (config/ntfy.env)")
        return
    url = f"{settings.ntfy_url.rstrip('/')}/{settings.ntfy_topic}"
    req = urllib.request.Request(url, data=msg.encode(), headers={"Title": "llm-bench"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"pushed: {msg}")
    except OSError as e:
        print(f"ntfy push failed: {e}")


def main() -> None:
    """Send argv[1] as ntfy message."""
    notify(" ".join(sys.argv[1:]) or "llm-bench fertig")


if __name__ == "__main__":
    main()
