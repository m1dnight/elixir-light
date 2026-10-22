"""
Build a PrusaSlicer 3MF project from a flat-colored SVG logo.

The logo is sliced into shades (see slice_svg.py). The full logo becomes a
white base part, and the shades become one relief part on top of it, where
every shade is extruded to a thickness that depends on how dark it is: the
darker the shade, the thicker the colored plastic and the less light gets
through. With --separate-shades every shade is its own part instead. Optionally a
clear part fills everything up to a total thickness and adds a border around
the logo, so the whole thing is one flat plate that can sit in the housing.

The plate is meant to be printed face down: the visible side is the first
layer (the white base), so it is mirrored (like the old STL files were). Parts
are already positioned, and the extruder for every part is set in the project,
so the file only has to be loaded in PrusaSlicer.

Usage: python build_3mf.py logos/elixir.svg
"""
import argparse
import pathlib
import zipfile

import numpy as np
import trimesh
from manifold3d import CrossSection, FillRule, Manifold
from shapely import affinity
from shapely.ops import unary_union

from slice_svg import full_logo, slice_svg

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def extrude(geometry, z_from, z_to):
    """Extrude a (multi)polygon between two heights into a single mesh."""
    polys = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    meshes = [trimesh.creation.extrude_polygon(p, z_to - z_from) for p in polys if p.area > 0.01]
    mesh = trimesh.util.concatenate(meshes)
    mesh.apply_translation([0, 0, z_from])
    mesh.fix_normals()
    return mesh


def cross_section(geometry):
    """A manifold3d CrossSection from a shapely (multi)polygon."""
    polys = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    rings = [np.array(ring.coords[:-1]) for p in polys if p.area > 0.01 for ring in [p.exterior, *p.interiors]]
    return CrossSection(rings, FillRule.EvenOdd)


def stack(regions, layer_counts, z_from, layer_height):
    """
    All shades as one solid: a stack of slabs, one per printed layer, where
    slab k covers every shade that is at least k layers thick. One part means
    the slicer never sees a boundary between shades, so no walls end up inside.
    """
    solid = None
    for k in range(1, max(layer_counts) + 1):
        # Neighbouring shades leave hairline gaps where their sampled edges
        # don't match exactly; buffer out and back in to close them.
        region = unary_union([r for r, n in zip(regions, layer_counts) if n >= k]).buffer(0.05).buffer(-0.05)
        slab = Manifold.extrude(cross_section(region), layer_height).translate([0, 0, z_from + (k - 1) * layer_height])
        solid = slab if solid is None else solid + slab
    mesh = solid.to_mesh()
    # manifold3d's output is already clean; trimesh's vertex merging would only break it.
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)


def build_parts(layers, width, height, args):
    """
    Turn the sliced shades into positioned meshes.

    Returns a list of (name, mesh, extruder) in print order.
    """
    scale = args.height / height
    thickness = [n * args.layer_height for n in args.layers]
    if len(thickness) < len(layers):
        raise SystemExit(f"logo has {len(layers)} shades but only {len(thickness)} layer counts were given")

    def place(geometry):
        # SVG y points down, so flip it to make the logo upright. Mirroring x
        # on top of that makes the logo read correctly when printed face down.
        xfact = -scale if args.mirror else scale
        geometry = affinity.scale(geometry, xfact=xfact, yfact=-scale, origin=(0, 0))
        return affinity.translate(geometry, xoff=width * scale if args.mirror else 0, yoff=height * scale)

    # Layer 0: the full logo as a white base, every shade stacks on top of it.
    logo = place(full_logo(layers))
    base = args.base_layers * args.layer_height
    parts = [("layer_0_white", extrude(logo, 0, base), args.white_extruder)]
    if args.separate_shades:
        for i, (color, region) in enumerate(layers):
            # Neighbouring shades must overlap a little. If they only touch, the
            # slicer sees a hairline gap, treats them as separate islands and puts
            # a wall around each; overlapping parts of the same material are merged.
            region = place(region).buffer(args.overlap).intersection(logo)
            parts.append((f"layer_{i + 1}_{color}", extrude(region, base, base + thickness[i]), args.color_extruder))
    else:
        regions = [place(region) for _, region in layers]
        parts.append(("shades", stack(regions, args.layers, base, args.layer_height), args.color_extruder))

    if args.total > 0:
        # The clear part: a border ring over the full height, and on top of every
        # shade whatever is left between the colored plastic and the total thickness.
        if args.total < base + max(thickness):
            raise SystemExit(f"--total {args.total} is thinner than the base plus the thickest shade")
        clear = [extrude(logo.buffer(args.border).difference(logo), 0, args.total)]
        for i, (_, region) in enumerate(layers):
            clear.append(extrude(place(region), base + thickness[i], args.total))
        parts.append(("clear", trimesh.util.concatenate(clear), args.clear_extruder))

    return parts


def write_3mf(parts, out_path, name):
    """Write parts as volumes of one object, PrusaSlicer style."""
    vertices, triangles, volumes = [], [], []
    for part_name, mesh, extruder in parts:
        first = len(triangles)
        triangles.extend(mesh.faces + len(vertices))
        vertices.extend(mesh.vertices)
        volumes.append((part_name, extruder, first, len(triangles) - 1))

    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
             f' <metadata name="Title">{name}</metadata>',
             ' <resources>', '  <object id="1" type="model">', '   <mesh>', '    <vertices>']
    model.extend(f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in vertices)
    model.append('    </vertices>')
    model.append('    <triangles>')
    model.extend(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in triangles)
    model.extend(['    </triangles>', '   </mesh>', '  </object>', ' </resources>',
                  ' <build>', '  <item objectid="1" printable="1"/>', ' </build>', '</model>'])

    config = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>', ' <object id="1" instances_count="1">',
              f'  <metadata type="object" key="name" value="{name}"/>']
    for part_name, extruder, first, last in volumes:
        config.extend([f'  <volume firstid="{first}" lastid="{last}">',
                       f'   <metadata type="volume" key="name" value="{part_name}"/>',
                       '   <metadata type="volume" key="volume_type" value="ModelPart"/>',
                       f'   <metadata type="volume" key="extruder" value="{extruder}"/>',
                       '  </volume>'])
    config.extend([' </object>', '</config>'])

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("3D/3dmodel.model", "\n".join(model))
        zf.writestr("Metadata/Slic3r_PE_model.config", "\n".join(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", nargs="?", default="logos/elixir.svg")
    parser.add_argument("--height", type=float, default=118.0, help="height of the logo in mm (default: 118)")
    parser.add_argument("--layer-height", type=float, default=0.1, help="layer height of the print in mm (default: 0.1)")
    parser.add_argument("--base-layers", type=int, default=4, help="number of white layers for the base (default: 4)")
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 8, 6, 4, 2],
                        help="number of colored layers per shade, darkest first (default: 10 8 6 4 2)")
    parser.add_argument("--total", type=float, default=0,
                        help="fill up to this thickness in mm with a clear part and add a border (default: off)")
    parser.add_argument("--border", type=float, default=1.0, help="clear border around the logo in mm (default: 1.0)")
    parser.add_argument("--separate-shades", action="store_true",
                        help="one part per shade instead of a single part, e.g. to print shades in different filaments")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="overlap between neighbouring shades in mm with --separate-shades (default: 0.1)")
    parser.add_argument("--white-extruder", type=int, default=1, help="extruder for the white base (default: 1)")
    parser.add_argument("--color-extruder", type=int, default=2, help="extruder for the colored parts (default: 2)")
    parser.add_argument("--clear-extruder", type=int, default=3, help="extruder for the clear part (default: 3)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="do not mirror the logo")
    args = parser.parse_args()

    svg_path = pathlib.Path(args.svg)
    layers, width, height = slice_svg(svg_path)
    parts = build_parts(layers, width, height, args)

    out_dir = svg_path.parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{svg_path.stem}.3mf"
    write_3mf(parts, out_path, svg_path.stem)

    for part_name, mesh, extruder in parts:
        lo, hi = mesh.bounds
        print(f"{part_name:>18}: extruder {extruder}, z {lo[2]:.1f}..{hi[2]:.1f} mm, "
              f"{len(mesh.faces)} triangles, watertight={mesh.is_watertight}")
    size = np.ptp(np.vstack([m.vertices for _, m, _ in parts]), axis=0)
    print(f"plate: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm -> {out_path}")


if __name__ == "__main__":
    main()
