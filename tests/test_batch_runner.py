"""Unit tests for globally phased trident.batch.run_slide_batch callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trident.batch.runner import run_slide_batch
from trident.batch.types import PipelineConfig, SlideEntry


def _entry(name: str, idx: int = 0) -> SlideEntry:
    return SlideEntry(slide_name=name, local_input_name=name, idx=idx)


@pytest.fixture
def pipeline_config(tmp_path):
    return PipelineConfig(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        target_mag=20,
        patch_size=256,
        overlap=0,
        patch_encoder="hoptimus0",
        segment_backend="otsu",
        gpu_id=-1,
    )


def test_run_slide_batch_global_phasing_and_callbacks(pipeline_config):
  entries = [_entry("slide_a.svs", 0), _entry("slide_b.svs", 1)]
  phase_starts: list[tuple[str, int]] = []
  progress: list[tuple[str, int, int]] = []
  item_done_phases: list[str] = []
  call_order: list[str] = []

  def on_phase_start(phase: str, total: int) -> None:
      phase_starts.append((phase, total))
      call_order.append(f"start:{phase}")

  def on_progress(phase: str, completed: int, total: int) -> None:
      progress.append((phase, completed, total))
      call_order.append(f"progress:{phase}:{completed}")

  def on_item_done(payload: dict) -> None:
      item_done_phases.append(str(payload.get("phase")))
      call_order.append(f"item:{payload.get('phase')}:{payload.get('idx')}")

  with patch("trident.batch.runner._run_segment") as mock_seg, patch(
      "trident.batch.runner._run_patch"
  ) as mock_patch, patch("trident.batch.runner._run_extract") as mock_extract, patch(
      "trident.batch.runner.cached_segmenter", return_value=MagicMock()
  ), patch(
      "trident.batch.runner.cached_patch_encoder", return_value=MagicMock()
  ), patch(
      "trident.batch.runner._resolve_device", return_value="cpu"
  ):
      mock_seg.side_effect = lambda *a, **k: call_order.append(f"seg:{a[0].slide_name}")
      mock_patch.side_effect = lambda *a, **k: call_order.append(f"patch:{a[0].slide_name}")
      mock_extract.side_effect = lambda *a, **k: call_order.append(f"extract:{a[0].slide_name}")

      code, results = run_slide_batch(
          entries,
          pipeline_config,
          stage="all",
          on_phase_start=on_phase_start,
          on_progress=on_progress,
          on_item_done=on_item_done,
      )

  assert code == 0
  assert len(results) == 2
  assert phase_starts == [("segment", 2), ("patch", 2), ("extract", 2)]
  assert progress[0] == ("segment", 1, 2)
  assert progress[1] == ("segment", 2, 2)
  assert item_done_phases[:2] == ["segment", "segment"]
  assert "patch" not in item_done_phases[:2]
  seg_indices = [i for i, token in enumerate(call_order) if token.startswith("seg:")]
  first_patch = next(i for i, token in enumerate(call_order) if token.startswith("patch:"))
  assert max(seg_indices) < first_patch
