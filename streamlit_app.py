"""
Entry point for Streamlit Cloud deployment.
Runs the dashboard from dashboard/app.py with correct path resolution.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
