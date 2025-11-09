import numpy as np
from stl import mesh


def flatten_stl_solid(input_stl_path, output_stl_path, base_height=0.0, top_height=1.0, remove_flat=True):
    """
    Flatten an STL by duplicating all triangles at two heights.
    This creates a dense "solid" appearance even though triangles may overlap.
    """
    # Load the STL file
    original_mesh = mesh.Mesh.from_file(input_stl_path)

    if remove_flat:
        vertices = original_mesh.vectors
        z_coords = vertices[:, :, 2]
        z_min = np.min(z_coords, axis=1)
        z_max = np.max(z_coords, axis=1)
        heights = z_max - z_min

        epsilon = 1e-6
        non_flat_mask = heights > epsilon

        print(f"Original triangles: {len(original_mesh.vectors)}")
        print(f"Flat triangles removed: {np.sum(~non_flat_mask)}")
        print(f"Remaining triangles: {np.sum(non_flat_mask)}")

        if np.sum(non_flat_mask) == 0:
            print("Error: All triangles are flat!")
            return None

        filtered_mesh = mesh.Mesh(original_mesh.data[non_flat_mask].copy())
    else:
        filtered_mesh = mesh.Mesh(original_mesh.data.copy())

    # Create arrays to hold all new triangles
    num_triangles = len(filtered_mesh.vectors)

    # We'll create: bottom layer + top layer + side walls for each triangle
    # Each triangle has 3 edges, so 3 side walls (each wall = 2 triangles)
    total_new_triangles = num_triangles * 2 + (num_triangles * 3 * 2)

    # Create new mesh data array
    new_data = np.zeros(total_new_triangles, dtype=mesh.Mesh.dtype)

    idx = 0

    # Add bottom layer (with flipped normals)
    bottom_triangles = filtered_mesh.data.copy()
    bottom_triangles['vectors'][:, :, 2] = base_height
    bottom_triangles['vectors'][:, [1, 2]] = bottom_triangles['vectors'][:, [2, 1]]
    new_data[idx:idx + num_triangles] = bottom_triangles
    idx += num_triangles

    # Add top layer
    top_triangles = filtered_mesh.data.copy()
    top_triangles['vectors'][:, :, 2] = top_height
    new_data[idx:idx + num_triangles] = top_triangles
    idx += num_triangles

    # Add side walls for each triangle
    for i, tri in enumerate(filtered_mesh.vectors):
        # Get the three vertices (flattened to XY)
        v0 = np.array([tri[0, 0], tri[0, 1]])
        v1 = np.array([tri[1, 0], tri[1, 1]])
        v2 = np.array([tri[2, 0], tri[2, 1]])

        # Create 3 vertical walls (one for each edge)
        edges = [(v0, v1), (v1, v2), (v2, v0)]

        for v_start, v_end in edges:
            # Create two triangles for this vertical wall
            # Triangle 1
            new_data['vectors'][idx] = [
                [v_start[0], v_start[1], base_height],
                [v_end[0], v_end[1], base_height],
                [v_start[0], v_start[1], top_height]
            ]
            idx += 1

            # Triangle 2
            new_data['vectors'][idx] = [
                [v_end[0], v_end[1], base_height],
                [v_end[0], v_end[1], top_height],
                [v_start[0], v_start[1], top_height]
            ]
            idx += 1

    # Create final mesh
    final_mesh = mesh.Mesh(new_data)
    final_mesh.update_normals()

    # Save
    final_mesh.save(output_stl_path)

    print(f"\nFlattened STL saved to {output_stl_path}")
    print(f"Final number of triangles: {len(final_mesh.vectors)}")

    return final_mesh


# Example usage:
if __name__ == "__main__":
    input_file = "output/elixir.stl"
    output_file = "output/flattened_elixir.stl"

    flattened = flatten_stl_solid(
        input_file,
        output_file,
        base_height=0.0,
        top_height=1.0,
        remove_flat=True
    )