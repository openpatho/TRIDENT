"""Pytest fixtures — stub optional heavy deps for lightweight unit tests."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

if "openslide" not in sys.modules:
    openslide_stub = types.ModuleType("openslide")
    openslide_stub.OpenSlide = MagicMock(name="OpenSlide")
    openslide_stub.OpenSlideError = Exception
    sys.modules["openslide"] = openslide_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn.functional = torch_nn_functional
    torch_stub.nn = torch_nn
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = MagicMock(name="DataLoader")
    torch_utils.data = torch_utils_data
    torch_stub.utils = torch_utils
    cuda_stub = MagicMock()
    cuda_stub.is_available = lambda: False
    torch_stub.cuda = cuda_stub
    torch_stub.device = MagicMock()
    torch_stub.Tensor = MagicMock()
    torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

if "torchvision" not in sys.modules:
    tv_stub = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.Compose = MagicMock()
    tv_stub.transforms = tv_transforms
    sys.modules["torchvision"] = tv_stub
    sys.modules["torchvision.transforms"] = tv_transforms

if "timm" not in sys.modules:
    timm_stub = types.ModuleType("timm")
    timm_stub.create_model = MagicMock()
    sys.modules["timm"] = timm_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = MagicMock()
    sys.modules["tqdm"] = tqdm_stub

if "geopandas" not in sys.modules:
    gpd_stub = types.ModuleType("geopandas")
    gpd_stub.GeoDataFrame = MagicMock()
    gpd_stub.read_file = MagicMock()
    gpd_stub.gpd = gpd_stub
    sys.modules["geopandas"] = gpd_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2_stub

if "h5py" not in sys.modules:
    h5py_stub = types.ModuleType("h5py")
    h5py_stub.File = MagicMock()
    sys.modules["h5py"] = h5py_stub

if "shapely" not in sys.modules:
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry = types.ModuleType("shapely.geometry")
    shapely_geometry.Polygon = MagicMock()
    shapely_stub.geometry = shapely_geometry
    shapely_stub.Polygon = shapely_geometry.Polygon
    sys.modules["shapely"] = shapely_stub
    sys.modules["shapely.geometry"] = shapely_geometry

if "pandas" not in sys.modules:
    pd_stub = types.ModuleType("pandas")
    pd_stub.DataFrame = MagicMock()
    sys.modules["pandas"] = pd_stub
