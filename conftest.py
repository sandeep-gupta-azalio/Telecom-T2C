"""Root pytest conftest: makes `from src import ...` resolve regardless of cwd.

Pytest auto-executes this before collecting tests and adds this file's
directory (the project root) to sys.path, since there's no __init__.py here.
The explicit insert below is a belt-and-suspenders guarantee that works the
same way whether pytest is invoked from the project root or from tests/.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
