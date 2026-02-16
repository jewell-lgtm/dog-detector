import subprocess
import time


class ScriptRunner:
    def __init__(self, cooldown: float = 5.0):
        self.cooldown = cooldown
        self._last_fired: dict[str, float] = {}
        self.total_fires: int = 0
        self.last_fired_path: str | None = None
        self.last_fired_time: float = 0.0

    def run(self, script_path: str) -> bool:
        """Run an AppleScript. Returns True if fired, False if on cooldown."""
        if not script_path:
            return False
        now = time.time()
        last = self._last_fired.get(script_path, 0.0)
        if now - last < self.cooldown:
            return False
        self._last_fired[script_path] = now
        self.last_fired_path = script_path
        self.last_fired_time = now
        self.total_fires += 1
        subprocess.Popen(["osascript", script_path])
        return True

    def cooldown_remaining(self, script_path: str) -> float:
        if not script_path:
            return 0.0
        elapsed = time.time() - self._last_fired.get(script_path, 0.0)
        return max(0.0, self.cooldown - elapsed)
