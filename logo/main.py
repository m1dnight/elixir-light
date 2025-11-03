import pathlib

import numpy as np
from PIL import Image
from PIL import ImageFilter
from stl import mesh
import pymesh

distinct_colors = 14
max_width = 1024
# layer_height = 0.2 # 0.49
layer_height = 0.1 # 0.22
transparent_layers = 5



def resize_image(image, max_size):
    # Get current dimensions
    width, height = image.size

    # Calculate the scaling factor
    # We want the larger dimension to be max_size
    if width > height:
        # Width is the limiting dimension
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        # Height is the limiting dimension
        new_height = max_size
        new_width = int((width * max_size) / height)

    # Resize the image
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_img


def remove_alpha(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image.convert('RGB')
    return new_image


def image_path(logo_path, suffix):
    filename = pathlib.Path(logo_path).stem
    no_alpha_filename = f"{filename}_{suffix}.png"
    return pathlib.Path(logo_path).parent / no_alpha_filename


def stl_path(logo_path, silhouette=False):
    filename = pathlib.Path(logo_path).stem
    stl_filename = f"{filename}.stl"
    if silhouette:
        stl_filename = f"{filename}_silhouette.stl"
    return pathlib.Path(logo_path).parent / stl_filename


def process_logo(logo_path):
    image = Image.open(logo_path)
    # resize image if necessary
    image = resize_image(image, max_width)

    # remove alpha if there is any
    image = remove_alpha(image)
    image.save(image_path(logo_path, "no_alpha"))

    # grayscale the image
    image = image.convert("L")
    image.save(image_path(logo_path, "grayscale"))

    # quantize the image
    image = image.quantize(distinct_colors)
    image.save(image_path(logo_path, "quantized"))

    # median filter to remove details
    image = image.convert("RGB")
    image = image.filter(ImageFilter.MedianFilter(size = 13))
    image = image.convert("L")
    image.save(image_path(logo_path, "median_filter"))

    # save final image
    image.save(image_path(logo_path, "final"))
    return image

def fill_holes(image_arr):
    # dimensions of the image
    height, width = image_arr.shape

    # get the second-lightest color in the image (lightest is white)
    lightest_color = sorted(np.unique(image_arr))[-2]
    # if there are holes in the image surrounded by colors, fill them in with white.
    # [1, 2, 3, 4, 5][1:-1]
    # drop first and last element
    for x in range(1, height - 1):
        row = image_arr[x]
        # all pixels that are going to be printed are not equal to 0 (no layers)
        indices = np.where(row != 0)[0]

        # if this row is entirely white,
        if indices.size < 2 or indices.size == width:
            continue
        for y in range(indices[0], indices[-1] + 1):
            if image_arr[x, y] == 0:
                image_arr[x, y] = 1

    return image_arr

def transparent_layer(image_arr):
    height, width = image_arr.shape

    def pixel_to_layer_count(pixel):
        if pixel != 0:
            return pixel + transparent_layers
        else:
            return pixel

    pixel_to_height_vec = np.vectorize(pixel_to_layer_count)

    return pixel_to_height_vec(image_arr)

def silhouette(image_arr):
    height, width = image_arr.shape

    def pixel_to_layer_count(pixel):
        if pixel != 0:
            return 0
        else:
            return 255

    pixel_to_height_vec = np.vectorize(pixel_to_layer_count)

    return pixel_to_height_vec(image_arr)

def image_to_heightmap(image_arr):
    # the image has grayscale values, so we clamp them.
    # darkest colors are lowest, brightest is highest.

    # dimensions of the image
    height, width = image_arr.shape

    # each pixel will become a point in a three-dimensional
    # space so the x and y are the x and y in the image,
    # the Z is computed based off of the brightness of the
    # pixel
    # list of points in the 3d space (x, y, z)
    def pixel_to_layer_count(pixel):
        layer_interval = 255 / distinct_colors

        # to get the lightest color to have the most layers, invert the value
        # add 1 for a fully transparent layer at the bottom
        # this layer must be set explicitly in the slicer.
        total_layers = (distinct_colors - int(pixel / layer_interval))
        return total_layers  # return int(layer_count)

    pixel_to_height_vec = np.vectorize(pixel_to_layer_count)

    return pixel_to_height_vec(image_arr)


def heightmap_to_stl(height_arr):
    # dimensions of the image
    height, width = height_arr.shape

    # convert the height array into a list of vertices
    vertices = []
    # x = row
    # y = col
    for y in range(height):
        for x in range(width):
            height_in_layers = height_arr[y][x] * layer_height
            vertices.append([x, y, height_in_layers])

    # add vertices for the bottom (now the heightmap is a hollow)
    for y in range(height):
        for x in range(width):
            vertices.append([x, y, 0])

    vertices = np.array(vertices)
    # create triangles for the mesh

    # the array look something like this:
    # (0, 0) - (1, 0) - (2, 0)
    #   |        |        |
    # (0, 1) - (1, 1) - (2, 1)
    #   |        |        |
    # (0, 2) - (1, 2) - (2, 2)
    #
    # point 00 becomes v1
    # point 10 becomes v2
    # point 01 becomes v3
    # the other triangle consists of
    # 01, 11 and 10.
    # below we compute the indices of these points as if the array was flat
    # v1 ---- v2
    # |    /  |
    # |   /   |
    # |  /    |
    # v3 ---- v4
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            # compute the index in the array as if the 2d array was a long flat array
            v1 = y * width + x
            v2 = y * width + (x + 1)
            v3 = (y + 1) * width + x
            v4 = (y + 1) * width + (x + 1)

            # # if the triangle is between all 0 points, discard it
            # # THIS WILL REMOVE ALL SURFACES EXCEPT THE TOP ONE SO ITS NOT CORRECT YET
            # if vertices[v1][2] != 0 and vertices[v2][2] != 0 and vertices[v3][2] != 0:
            faces.append([v1, v2, v3])
            # if vertices[v2][2] != 0 and vertices[v4][2] != 0 and vertices[v3][2] != 0:
            faces.append([v2, v4, v3])

    # these vertices are added after the vertices for the image (so an array of same length added)
    # | vertices for image | vertices for bottom |
    offset = width * height
    for y in range(height - 1):
        for x in range(width - 1):
            v1 = offset + y * width + x
            v2 = offset + y * width + (x + 1)
            v3 = offset + (y + 1) * width + x
            v4 = offset + (y + 1) * width + (x + 1)

            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    for y in range(height - 1):
        top1 = y * width + 0
        top2 = (y + 1) * width + 0
        bot1 = offset + y * width + 0
        bot2 = offset + (y + 1) * width + 0

        faces.append([top1, bot1, top2])
        faces.append([bot1, bot2, top2])

    # Right wall (x = width-1)
    for y in range(height - 1):
        top1 = y * width + (width - 1)
        top2 = (y + 1) * width + (width - 1)
        bot1 = offset + y * width + (width - 1)
        bot2 = offset + (y + 1) * width + (width - 1)

        faces.append([top1, top2, bot1])
        faces.append([bot1, top2, bot2])

    # Front wall (y = 0)
    for x in range(width - 1):
        top1 = 0 * width + x
        top2 = 0 * width + (x + 1)
        bot1 = offset + 0 * width + x
        bot2 = offset + 0 * width + (x + 1)

        faces.append([top1, top2, bot1])
        faces.append([bot1, top2, bot2])

    # Back wall (y = height-1)
    for x in range(width - 1):
        top1 = (height - 1) * width + x
        top2 = (height - 1) * width + (x + 1)
        bot1 = offset + (height - 1) * width + x
        bot2 = offset + (height - 1) * width + (x + 1)

        faces.append([top1, bot1, top2])
        faces.append([bot1, bot2, top2])

    faces = np.array(faces)

    # Create mesh object
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j], :]

    return stl_mesh


def remove_zero_height(stl_mesh, base_height=0):
    """Remove faces that are at or below base_height"""
    valid_faces = []

    for i, triangle in enumerate(stl_mesh.vectors):
        max_z = np.max(triangle[:, 2])  # Z coordinates

        if max_z > base_height + 0.01:  # Small tolerance
            valid_faces.append(i)

    # Create new mesh with only valid faces
    if len(valid_faces) > 0:
        new_vectors = stl_mesh.vectors[valid_faces]
        new_mesh = mesh.Mesh(np.zeros(len(new_vectors), dtype=mesh.Mesh.dtype))
        new_mesh.vectors = new_vectors
        return new_mesh
    else:
        return None

def save_stl_mesh(stl_mesh, stl_path):
    # Save to file
    stl_mesh.save(stl_path)


def debugprint(height_arr):
    layer_heights = np.unique(height_arr)
    minimal_layer_height = np.min(layer_heights)
    maximal_layer_height = np.max(layer_heights)
    expected_height = maximal_layer_height * layer_height
    print("""
    Lowest layer   : {}
    Highest layer  : {}
    Layers         : {}
    Layer height   : {}
    Expected height: {}
    """.format(minimal_layer_height, maximal_layer_height, layer_heights, layer_height, expected_height))

logo = "logos/elixir.png"
image = process_logo(logo)
image_arr = np.array(image)

height_arr = image_to_heightmap(image_arr)
height_arr = fill_holes(height_arr)
height_arr = transparent_layer(height_arr)
debugprint(height_arr)
stl_mesh = heightmap_to_stl(height_arr)
save_stl_mesh(stl_mesh, stl_path(logo))
