from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trident")
except PackageNotFoundError:
    __version__ = "unknown"

_EXPORTS = {
    "PipelineConfig": ("trident.batch.types", "PipelineConfig"),
    "SlideEntry": ("trident.batch.types", "SlideEntry"),
    "prewarm_models": ("trident.batch.cache", "prewarm_models"),
    "run_slide_batch": ("trident.batch.runner", "run_slide_batch"),
    "Processor": ("trident.Processor", "Processor"),
    "load_wsi": ("trident.wsi_objects.WSIFactory", "load_wsi"),
    "OpenSlideWSI": ("trident.wsi_objects.OpenSlideWSI", "OpenSlideWSI"),
    "ImageWSI": ("trident.wsi_objects.ImageWSI", "ImageWSI"),
    "CuCIMWSI": ("trident.wsi_objects.CuCIMWSI", "CuCIMWSI"),
    "SDPCWSI": ("trident.wsi_objects.SDPCWSI", "SDPCWSI"),
    "OMEZarrWSI": ("trident.wsi_objects.OMEZarrWSI", "OMEZarrWSI"),
    "CZIWSI": ("trident.wsi_objects.CZIWSI", "CZIWSI"),
    "WSIPatcher": ("trident.wsi_objects.WSIPatcher", "WSIPatcher"),
    "OpenSlideWSIPatcher": ("trident.wsi_objects.WSIPatcher", "OpenSlideWSIPatcher"),
    "WSIPatcherDataset": ("trident.wsi_objects.WSIPatcherDataset", "WSIPatcherDataset"),
    "visualize_heatmap": ("trident.Visualization", "visualize_heatmap"),
    "AnyToTiffConverter": ("trident.Converter", "AnyToTiffConverter"),
    "deprecated": ("trident.Maintenance", "deprecated"),
    "WSIReaderType": ("trident.wsi_objects.WSIFactory", "WSIReaderType"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS.keys()) + ["__version__"]
