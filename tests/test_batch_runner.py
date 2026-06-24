"""Unit tests for trident batch runner and model cache."""

from __future__ import annotations

from unittest import mock

from trident.batch.types import PipelineConfig, SlideEntry


def test_expand_stages_segment_patch():
    from trident.batch.runner import _expand_stages

    assert _expand_stages("segment_patch") == ["segment", "patch"]
    assert _expand_stages("all") == ["segment", "patch", "extract"]


def test_run_slide_batch_callbacks_order():
    from trident.batch import run_slide_batch

    entries = [
        SlideEntry(slide_name="a.svs", local_input_name="a.svs"),
        SlideEntry(slide_name="b.svs", local_input_name="b.svs"),
    ]
    config = PipelineConfig(
        input_dir="/in",
        output_dir="/out",
        gpu_id=0,
        segment_backend="hest",
    )
    progress_events: list[tuple] = []
    item_events: list[dict] = []

    with (
        mock.patch("trident.batch.runner._resolve_device", return_value="cuda:0"),
        mock.patch("trident.batch.runner._run_segment") as seg,
        mock.patch("trident.batch.runner._run_patch") as patch,
        mock.patch("trident.batch.runner._run_extract") as extract,
        mock.patch("trident.batch.runner.cached_segmenter", return_value=mock.Mock()),
        mock.patch("trident.batch.runner.cached_patch_encoder", return_value=mock.Mock()),
    ):
        code, results = run_slide_batch(
            entries,
            config,
            stage="segment",
            on_progress=lambda phase, n, t: progress_events.append((phase, n, t)),
            on_item_done=item_events.append,
        )

    assert code == 0
    assert seg.call_count == 2
    patch.assert_not_called()
    extract.assert_not_called()
    assert len(item_events) == 2
    assert progress_events[-1] == ("segment", 2, 2)


def test_model_cache_reuse():
    import sys
    import types

    from trident.batch.cache import cached_patch_encoder, _MODEL_CACHE

    _MODEL_CACHE.clear()
    factory = mock.Mock(return_value=mock.Mock(enc_name="hoptimus0"))
    stub = types.ModuleType("trident.patch_encoder_models.load")
    stub.encoder_factory = factory
    sys.modules["trident.patch_encoder_models.load"] = stub
    try:
        a = cached_patch_encoder("hoptimus0", "cpu")
        b = cached_patch_encoder("hoptimus0", "cpu")
    finally:
        sys.modules.pop("trident.patch_encoder_models.load", None)
    assert a is b
    assert factory.call_count == 1


def test_failed_slide_skips_later_stages():
    from trident.batch import run_slide_batch

    entries = [
        SlideEntry(slide_name="bad.svs", local_input_name="bad.svs"),
        SlideEntry(slide_name="good.svs", local_input_name="good.svs"),
    ]
    config = PipelineConfig(input_dir="/in", output_dir="/out", gpu_id=0, segment_backend="hest")
    patch_calls: list[str] = []

    def _segment(entry, *_args, **_kwargs):
        if entry.slide_name == "bad.svs":
            raise RuntimeError("segment failed")

    def _patch(entry, *_args, **_kwargs):
        patch_calls.append(entry.slide_name)

    with (
        mock.patch("trident.batch.runner._resolve_device", return_value="cuda:0"),
        mock.patch("trident.batch.runner._run_segment", side_effect=_segment),
        mock.patch("trident.batch.runner._run_patch", side_effect=_patch),
        mock.patch("trident.batch.runner._run_extract"),
        mock.patch("trident.batch.runner.cached_segmenter", return_value=mock.Mock()),
        mock.patch("trident.batch.runner.cached_patch_encoder", return_value=mock.Mock()),
    ):
        code, results = run_slide_batch(entries, config, stage="all")

    assert code == 0
    assert patch_calls == ["good.svs"]
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "ok"
