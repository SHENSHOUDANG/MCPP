from __future__ import annotations

import argparse
import io
import math
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_BBOX = (30.600, 122.010, 30.655, 122.100)  # south, west, north, east
TILE_SIZE = 256

PROVIDERS = {
    "arcgis_imagery": {
        "template": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "extension": "jpg",
    },
    "osm_standard": {
        "template": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "extension": "png",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real web-map tiles for Yangshan Port without model overlays.")
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="arcgis_imagery")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--south", type=float, default=DEFAULT_BBOX[0])
    parser.add_argument("--west", type=float, default=DEFAULT_BBOX[1])
    parser.add_argument("--north", type=float, default=DEFAULT_BBOX[2])
    parser.add_argument("--east", type=float, default=DEFAULT_BBOX[3])
    parser.add_argument("--cache-dir", default="outputs/real_map_tiles/cache")
    parser.add_argument("--output-dir", default="outputs/real_map_tiles")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    bbox = (args.south, args.west, args.north, args.east)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"yangshan_{args.provider}_z{args.zoom}.png"
    fetch_real_map(
        provider=args.provider,
        zoom=args.zoom,
        bbox=bbox,
        cache_dir=Path(args.cache_dir),
        output=output,
        timeout=args.timeout,
    )
    print(output.resolve())


def fetch_real_map(provider: str, zoom: int, bbox: tuple[float, float, float, float], cache_dir: Path, output: Path, timeout: float) -> Path:
    south, west, north, east = bbox
    x_min, y_north = lonlat_to_tile_float(west, north, zoom)
    x_max, y_south = lonlat_to_tile_float(east, south, zoom)
    tile_x0, tile_x1 = math.floor(x_min), math.floor(x_max)
    tile_y0, tile_y1 = math.floor(y_north), math.floor(y_south)

    provider_config = PROVIDERS[provider]
    cache_dir = cache_dir / provider / f"z{zoom}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    width_tiles = tile_x1 - tile_x0 + 1
    height_tiles = tile_y1 - tile_y0 + 1
    mosaic = Image.new("RGB", (width_tiles * TILE_SIZE, height_tiles * TILE_SIZE))

    for x in range(tile_x0, tile_x1 + 1):
        for y in range(tile_y0, tile_y1 + 1):
            tile = _load_tile(provider_config["template"], provider_config["extension"], zoom, x, y, cache_dir, timeout)
            mosaic.paste(tile.convert("RGB"), ((x - tile_x0) * TILE_SIZE, (y - tile_y0) * TILE_SIZE))
            time.sleep(0.05)

    left = int(round((x_min - tile_x0) * TILE_SIZE))
    right = int(round((x_max - tile_x0) * TILE_SIZE))
    top = int(round((y_north - tile_y0) * TILE_SIZE))
    bottom = int(round((y_south - tile_y0) * TILE_SIZE))
    crop = mosaic.crop((left, top, max(right, left + 1), max(bottom, top + 1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return output


def _load_tile(template: str, extension: str, zoom: int, x: int, y: int, cache_dir: Path, timeout: float) -> Image.Image:
    path = cache_dir / f"{x}_{y}.{extension}"
    if path.exists():
        return Image.open(path).copy()
    url = template.format(z=zoom, x=x, y=y)
    headers = {
        "User-Agent": "GAT-MAPPO-port-inspection-research/1.0 (local map preview)",
        "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*",
    }
    last_error: Exception | None = None
    part_path = path.with_suffix(path.suffix + ".part")
    curl_command = [
        "curl.exe",
        "-L",
        "--http1.1",
        "--fail",
        "--ssl-no-revoke",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        str(int(timeout)),
        "--max-time",
        str(int(timeout * 3)),
        "-A",
        "GAT-MAPPO-port-inspection-research/1.0",
        "-o",
        str(part_path),
        url,
    ]
    try:
        subprocess.run(curl_command, check=True)
        part_path.replace(path)
        return Image.open(path).copy()
    except Exception:
        if part_path.exists():
            part_path.unlink()

    for attempt in range(6):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
            path.write_bytes(data)
            return Image.open(io.BytesIO(data)).copy()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            time.sleep(0.8 + attempt * 0.8)
    raise RuntimeError(f"failed to fetch tile z={zoom} x={x} y={y}: {last_error}")


def lonlat_to_tile_float(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


if __name__ == "__main__":
    main()
