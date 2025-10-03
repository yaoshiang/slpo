import pytest


def test_import_dpo_trainer():
  """Sanity check test to ensure the vendored DPO library is importable."""
  try:
    from third_party.dpo.trainers import BasicTrainer
  except ImportError as e:
    pytest.fail(
      f"Failed to import BasicTrainer from the vendored DPO library: {e}"
    )
