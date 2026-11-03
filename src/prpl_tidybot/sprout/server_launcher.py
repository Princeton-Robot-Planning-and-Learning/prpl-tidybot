"""Launch main's arm/base RPC servers. Startup itself commands no motion."""

import signal
import socket
import subprocess
import sys
import time


def main() -> None:
    """Keep main's two servers alive in one laptop SSH terminal."""
    for port in (50000, 50001):
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"RPC port {port} is already in use. Use its existing launch "
                    "session, or stop that server in its own terminal before "
                    "starting the updated servers. No process was killed."
                )
    processes: list[subprocess.Popen] = []

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        for module in ("arm_server", "base_server"):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-u", "-m", "prpl_tidybot.third_party." + module]
                )
            )
        print(
            "Original PRPL arm/base RPC servers. Leave this terminal open.", flush=True
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
        raise RuntimeError("A PRPL server exited; inspect the output above")
    except KeyboardInterrupt:
        print("Stopping RPC servers launched by this command.", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
