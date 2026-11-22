"""GeoTIFF export (EPSG:4326, float32) without GDAL."""
from __future__ import annotations

import io

import numpy as np


def geotiff_bytes(arr: np.ndarray, bounds, nodata: float | None = None, description: str = "") -> bytes:
    import tifffile
    H, W = arr.shape
    lon0, lat0, lon1, lat1 = bounds
    sx, sy = (lon1 - lon0) / W, (lat1 - lat0) / H
    geokeys = [1, 1, 0, 3,
               1024, 0, 1, 2,      # GTModelTypeGeoKey = Geographic
               1025, 0, 1, 1,      # GTRasterTypeGeoKey = PixelIsArea
               2048, 0, 1, 4326]   # GeographicTypeGeoKey = WGS 84
    extratags = [
        (33550, "d", 3, (sx, sy, 0.0), False),                           # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, lon0, lat1, 0.0), False),        # ModelTiepoint
        (34735, "H", len(geokeys), tuple(geokeys), False),               # GeoKeyDirectory
    ]
    if nodata is not None:
        extratags.append((42113, "s", 0, str(nodata), False))           # GDAL_NODATA
    description = description.replace("²", "2").encode("ascii", "replace").decode()
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr.astype(np.float32), extratags=extratags, description=description,
                     compression="zlib", photometric="minisblack")
    return buf.getvalue()
