"""Entry point for ``python -m commerce_signals``.

Delegates to Kedro's CLI, same as ``kedro run`` from the shell.
"""

from kedro.framework.cli import main as kedro_main

kedro_main()
