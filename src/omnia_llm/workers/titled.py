"""Run an engine under a neutral process title.

The box is shared, and every user on it can read every process's command line (`ps`, and the
process column of `nvidia-smi`). Launched directly, the engines showed this project's full path
— and vLLM's showed the model too. The path grants nothing (the home directory is not readable by
anyone else), but it is nobody else's business either.

`setproctitle` rewrites the visible command line in place, which is what vLLM already does for
its own `VLLM::EngineCore` worker. Usage::

    python -m omnia_llm.workers.titled <title> <module> [args...]

``<module>`` must expose ``main()``; it runs with ``sys.argv = [<module>, *args]``.
"""

from __future__ import annotations

import importlib
import sys

from setproctitle import setproctitle


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python -m omnia_llm.workers.titled <title> <module> [args...]")
    title, module = sys.argv[1], sys.argv[2]
    setproctitle(title)
    sys.argv = [module, *sys.argv[3:]]
    sys.exit(importlib.import_module(module).main())


if __name__ == "__main__":
    main()
