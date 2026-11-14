"""Minimal runner (no pytest needed). Also works with: python -m pytest tests/
Use the analyzer's venv:  ~/.autocut/venv/bin/python tests/run_all.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

failed = 0
for modname in ("test_analyzer",):
    mod = __import__(modname)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)()
            print("PASS", modname, name)
        except Exception:
            failed += 1
            print("FAIL", modname, name)
            traceback.print_exc()
print("\n%s" % ("all passed" if not failed else "%d failed" % failed))
sys.exit(1 if failed else 0)
