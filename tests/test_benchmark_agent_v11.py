from pathlib import Path
import py_compile


def test_benchmark_agent_v11_is_valid_python() -> None:
    script = Path(__file__).resolve().parents[1] / "tools" / "benchmark_agent_v11.py"
    py_compile.compile(str(script), doraise=True)
