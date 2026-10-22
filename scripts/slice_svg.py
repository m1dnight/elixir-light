"""
Split a flat-colored SVG logo into one SVG per shade.

Every output SVG uses the same coordinates, so they line up when imported into
Fusion or a slicer. Layer 0 is the full logo (the base), layer 1 is the darkest
shade, layer 2 the next, etc. Paths that are painted over by later paths in the
SVG are clipped, so each layer only contains what is actually visible in the
original. A border SVG with the logo outline offset outwards is written too,
to build an enclosure from.

Usage: python slice_svg.py logos/elixir.svg --height 118
"""
import argparse
import io
import pathlib
import re

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from svgelements import SVG, Color, Path

# Sampling distance (in SVG units) used to flatten curves into polylines.
STEP = 0.1


def luminance(color):
    return 0.299 * color.red + 0.587 * color.green + 0.114 * color.blue


def path_to_polygon(path):
    """Flatten a path into a shapely polygon by sampling its curves."""
    polys = []
    for subpath in path.as_subpaths():
        points = []
        for segment in Path(subpath):
            if segment.length() == 0:
                continue
            n = max(2, int(segment.length() / STEP))
            points.extend(segment.point(t) for t in np.linspace(0, 1, n, endpoint=False))
        if len(points) >= 3:
            polys.append(Polygon([(p.x, p.y) for p in points]).buffer(0))
    return unary_union(polys)


def polygon_to_d(geometry):
    """Convert a (multi)polygon into SVG path data."""
    polys = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    d = []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = ["{:.3f} {:.3f}".format(x, y) for x, y in ring.coords[:-1]]
            d.append("M" + " L".join(coords) + " Z")
    return " ".join(d)


def slice_svg(svg_path):
    """
    Split an SVG into its visible regions per color.

    Returns:
        (layers, width, height) where layers is a list of (hex color, geometry)
        sorted from darkest to lightest, in SVG coordinates (y pointing down).
    """
    # svgelements does not understand css variables, keep the fallback value.
    text = re.sub(r"var\(--[\w-]+,\s*([^)]+)\)", r"\1", pathlib.Path(svg_path).read_text())
    svg = SVG.parse(io.StringIO(text), reify=True)

    # Paths in paint order, with their color and flattened geometry.
    paths = [(p.fill, path_to_polygon(p)) for p in svg.elements() if isinstance(p, Path) and p.fill]

    # Clip every path by everything painted on top of it, then merge by color.
    visible = {}
    for i, (color, geom) in enumerate(paths):
        above = unary_union([g for _, g in paths[i + 1:]])
        region = geom.difference(above)
        if not region.is_empty:
            visible.setdefault(color.hex, []).append(region)

    shades = sorted(visible, key=lambda c: luminance(Color(c)))
    layers = [(color, unary_union(visible[color]).simplify(STEP / 4)) for color in shades]
    return layers, svg.viewbox.width, svg.viewbox.height


def full_logo(layers):
    """
    The union of all shades: the outline of the whole logo.
    Adjacent shades leave hairline gaps where their sampled edges don't match
    exactly, so the union is buffered out and back in to close those.
    """
    return unary_union([region for _, region in layers]).buffer(STEP / 2).buffer(-STEP / 2).simplify(STEP / 4)


# Fusion (and browsers) read SVG user units as CSS pixels at 96 dpi and ignore
# the width/height attributes, so coordinates are written in px to come out at
# the right size in mm.
PX_PER_MM = 96 / 25.4


def write_svg(path, color, region, viewbox):
    """Write a region (in mm) as an SVG. viewbox is (x, y, width, height) in mm."""
    x, y, w, h = (v * PX_PER_MM for v in viewbox)
    region = affinity.scale(region, PX_PER_MM, PX_PER_MM, origin=(0, 0))
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.3f} {y:.3f} {w:.3f} {h:.3f}" '
        f'width="{viewbox[2]:.3f}mm" height="{viewbox[3]:.3f}mm">\n'
        f'  <path fill="{color}" fill-rule="evenodd" d="{polygon_to_d(region)}"/>\n'
        "</svg>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", nargs="?", default="logos/elixir.svg")
    parser.add_argument("--height", type=float, help="scale the output so the logo is this many mm high (default: keep SVG units)")
    parser.add_argument("--border", type=float, default=1.0, help="distance of the border outline around the logo in mm (default: 1.0)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="grow every shade by this many mm so hand-placed parts overlap instead of leaving gaps (default: 0)")
    args = parser.parse_args()

    svg_path = pathlib.Path(args.svg)
    layers, width, height = slice_svg(svg_path)
    base = full_logo(layers)

    # Scale everything so that one SVG unit is one millimetre.
    scale = args.height / height if args.height else 1.0
    width, height = width * scale, height * scale
    base = affinity.scale(base, scale, scale, origin=(0, 0))
    layers = [(color, affinity.scale(region, scale, scale, origin=(0, 0))) for color, region in layers]

    out_dir = svg_path.parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # Layer 0 is the full logo, the white base all other layers sit on.
    for layer, (color, region) in enumerate([("#ffffff", base), *layers]):
        if layer > 0 and args.overlap > 0:
            region = region.buffer(args.overlap).intersection(base)
        out = out_dir / f"{svg_path.stem}_layer_{layer}.svg"
        write_svg(out, color, region, (0, 0, width, height))
        print(f"layer {layer}: {color} luminance={luminance(Color(color)):.0f} area={region.area:.0f} -> {out}")

    # The border: a ring around the logo, in the same coordinates as the layers
    # so it lines up with them. Both its inner and outer edge are closed outlines.
    outer = base.buffer(args.border, join_style="round")
    out = out_dir / f"{svg_path.stem}_border.svg"
    write_svg(out, "#888888", outer.difference(base), (-args.border, -args.border, width + 2 * args.border, height + 2 * args.border))
    x0, y0, x1, y1 = outer.bounds
    print(f"border: {args.border}mm around the logo, outer outline {x1 - x0:.1f} x {y1 - y0:.1f} -> {out}")


if __name__ == "__main__":
    main()
