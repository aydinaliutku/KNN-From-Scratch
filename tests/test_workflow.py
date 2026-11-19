import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class KNNWorkflowTests(unittest.TestCase):
    def test_digits_cli_writes_metrics_and_confusion_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "evaluate.py", "--output", temporary],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((Path(temporary) / "metrics.json").read_text())
            self.assertEqual(metrics["test_examples"], 360)
            self.assertGreater(metrics["test_macro_f1"], 0.95)
            self.assertTrue((Path(temporary) / "confusion_matrix.png").is_file())

            notebook = subprocess.run(
                [sys.executable, "-c", """
import sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

path = Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
book = nbformat.read(path, as_version=4)
nbformat.validate(book)
shell = InteractiveShell.instance()
for cell in book.cells:
    if cell.cell_type == "code":
        shell.run_cell(cell.source).raise_error()
""", str(ROOT / "KNN_From_Scratch.ipynb")],
                cwd=temporary, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(notebook.returncode, 0, notebook.stderr)
            self.assertTrue((Path(temporary) / "outputs" / "confusion_matrix.png").is_file())
