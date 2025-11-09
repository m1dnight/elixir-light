import pathlib

import numpy as np
import openstl
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from stl import mesh


class ImageContainer:
    def __init__(self, image=None, image_path=None):
        self.image = image
        self.image_path = image_path

    @classmethod
    def from_image(cls, image_path):
        """
        Create an ImageContainer from an image file.

        Args:
            image_path: Path to the image file

        Returns:
            ImageContainer instance with the image as a numpy array
        """
        image = Image.open(image_path)
        return cls(image, image_path)

    def debugprint(self):
        """
        Print out some debug information about the image.
        """
        # unique values in the image array
        image_arr = np.array(self.image)
        unique_colors = sorted(np.unique(image_arr))
        print("unique colors: {}".format([int(x) for x in unique_colors]))
        print("min pixel value: {}, max pixel value: {}".format(np.min(image_arr), np.max(image_arr)))
        print("number of unique colors: {}".format(len(unique_colors)))

    def process_logo(self, scale=3, colors=14, median_size=13, border_width=10, fill_holes=True):
        """
        Process the logo through all transformation steps.

        Args:
            max_width: Maximum width for resizing (default: 256)
            max_height: Maximum height for resizing (default: 256)
            colors: Number of colors to quantize to (default: 14)
            median_size: Size of median filter kernel (default: 13)
            border_width: Width of border in pixels (default: 10)

        Returns:
            self (for method chaining)
        """
        image =       (self.resize(scale=scale).remove_alpha().grayscale().quantize(colors=colors).add_border(
            border_width=border_width).median_filter(size=median_size))

        if fill_holes:
            return image.fill_holes()
        else:
            return image

    def resize(self, scale=3, resample=Image.LANCZOS):
        """
        Resize the image to the specified dimensions.

        Args:
            max_width: Target width in pixels
            max_height: Target height in pixels
            resample: Resampling filter (default: LANCZOS for high quality)
                     Options: Image.NEAREST, Image.BOX, Image.BILINEAR,
                             Image.HAMMING, Image.BICUBIC, Image.LANCZOS

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Get current dimensions
        img_width, img_height = self.image.size
        # Resize the image
        self.image = self.image.resize((int(img_width * scale), int(img_height * scale)), Image.Resampling.LANCZOS)

        self.save("resized")
        return self

    def remove_alpha(self):
        """
        Remove the alpha channel from the image if present.
        If the image has no alpha channel, does nothing.

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Check if image has alpha channel
        if self.image.mode in ('RGBA', 'LA', 'PA'):
            # Create white background and paste image on top
            new_image = Image.new("RGBA", self.image.size, "WHITE")
            new_image.paste(self.image, (0, 0), self.image)
            self.image = new_image.convert('RGB')

        self.save("no_alpha")
        return self

    def grayscale(self):
        """
        Convert the image to grayscale.

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to grayscale
        self.image = self.image.convert("L")

        self.save("grayscale")
        return self

    def quantize(self, colors=14):
        """
        Quantize the image to a limited number of colors.

        Args:
            colors: Number of colors to quantize to (default: 14)

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Quantize the image
        self.image = self.image.quantize(colors)

        self.save("quantized")
        return self

    def median_filter(self, size=13):
        """
        Apply a median filter to the image to remove details.

        Args:
            size: Size of the median filter kernel (default: 13)

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert to RGB if needed (median filter works better with RGB)
        if self.image.mode != "RGB":
            self.image = self.image.convert("RGB")

        # Apply median filter
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))

        # Convert back to grayscale
        self.image = self.image.convert("L")

        self.save("median_filter")
        return self

    def add_border(self, border_width=10, fill='white'):
        """
        Add a border to the image.

        Args:
            border_width: Width of the border in pixels (default: 10)
            fill: Color to fill the border with (default: 'white')

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Add border using ImageOps.expand
        self.image = ImageOps.expand(self.image, border=border_width, fill=fill)

        self.save("border")
        return self

    def fill_holes(self):
        """
        Fill holes in the image that are surrounded by colors.
        Converts white pixels (255) that are between colored pixels to the lightest color.

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert image to numpy array
        image_arr = np.array(self.image)

        # dimensions of the image
        height, width = image_arr.shape

        # get the second-lightest color in the image (lightest is white, i.e., 255)
        lightest_color = sorted(np.unique(image_arr))[-2] if len(np.unique(image_arr)) > 1 else 255

        # if there are holes in the image surrounded by colors, fill them in
        for x in range(1, height - 1):
            row = image_arr[x]
            # all pixels that are going to be printed are not equal to 255 (white)
            indices = np.where(row != 255)[0]

            # if this row is entirely white or has less than 2 colored pixels, skip it
            if indices.size < 2 or indices.size == width:
                continue

            # fill white pixels between first and last colored pixel
            for y in range(indices[0], indices[-1] + 1):
                if image_arr[x, y] == 255:
                    image_arr[x, y] = lightest_color

        # Convert back to PIL Image
        self.image = Image.fromarray(image_arr)

        self.save("filled")
        return self

    def create_silhouette(self, border_width=10):
        """
        Create a silhouette from the image and save it.
        White pixels (255) stay white, all colored pixels become black (0).
        Adds a border around the non-white pixels by dilation.
        Does not modify the class's image.

        Args:
            border_width: Width of the border in pixels around the silhouette (default: 10)

        Returns:
            self (for method chaining)
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Convert image to numpy array
        image_arr = np.array(self.image)

        # Create silhouette: white stays white (255), everything else becomes black (0)
        silhouette_arr = np.where(image_arr == 255, 255, 0).astype(np.uint8)

        if border_width > 0:
            # Create mask of black pixels (non-white pixels)
            mask = silhouette_arr == 0

            # Manual dilation using sliding window to add border around black pixels
            padded = np.pad(mask, border_width, mode='constant', constant_values=False)
            dilated = np.zeros_like(mask)

            for i in range(-border_width, border_width + 1):
                for j in range(-border_width, border_width + 1):
                    dilated |= padded[border_width + i: border_width + i + mask.shape[0],
                               border_width + j: border_width + j + mask.shape[1]]

            # Border is dilated minus original (new pixels around the silhouette)
            border_mask = dilated & ~mask

            # Set border pixels to black as well
            silhouette_arr[border_mask] = 0

        # smooth the edges
        img = Image.fromarray(silhouette_arr)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        smoothed = blurred.point(lambda x: 0 if x > 127 else 255, mode='1')

        image_arr = np.array(smoothed)

        # Convert back: 1 -> 0 (black), 0 -> 255 (white)
        silhouette_arr = ((1 - image_arr) * 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(silhouette_arr)

        self.save("silhouette")
        return self

    def save(self, suffix):
        """
        Save the current image to a file with the given suffix.
        Creates a filename based on the original image path with the suffix appended.
        Files are saved to the output/ directory.

        Args:
            suffix: Suffix to append to the filename (e.g., "no_alpha", "grayscale")

        Returns:
            Path: The path where the image was saved
        """
        if self.image is None:
            raise ValueError("No image loaded")

        if self.image_path is None:
            raise ValueError("No image path set. Cannot determine where to save the image.")

        # Create the save path in output/ directory
        filename = pathlib.Path(self.image_path).stem
        save_filename = f"{filename}_{suffix}.png"

        # Get the base directory (parent of the image's parent directory)
        # e.g., if image is at "logos/elixir.png", go up to the base and then to "output/"
        base_dir = pathlib.Path(self.image_path).parent.parent
        output_dir = base_dir / "output"

        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        save_path = output_dir / save_filename

        # Save the image
        self.image.save(save_path)

        return save_path


class Heightmap:
    def __init__(self, image_container, distinct_colors=14):
        """
        Create a heightmap from an ImageContainer.

        Args:
            image_container: ImageContainer instance with a processed grayscale image
            distinct_colors: Number of distinct color layers (default: 14)
        """
        if image_container.image is None:
            raise ValueError("ImageContainer has no image loaded")

        self.distinct_colors = distinct_colors
        self.image_container = image_container

        # Convert PIL image to numpy array
        image_arr = np.array(image_container.image)

        # Convert image array to heightmap
        self.heightmap = self._image_to_heightmap(image_arr)

    def debugprint(self):
        # unique values in the image array
        unique_colors = sorted(np.unique(self.heightmap))
        print("unique heights: {}".format([x for x in unique_colors]))

    def _image_to_heightmap(self, image_arr):
        """
        Convert a grayscale image array to a heightmap.
        Darker colors = higher height, brighter colors = lower height.
        White (255) = 0 layers (background)

        Args:
            image_arr: Numpy array representing the grayscale image

        Returns:
            Numpy array with height values (layer counts)
        """

        def pixel_to_layer_count(pixel):
            # White pixels (255) should be 0 height (background)
            # Darker pixels should be taller
            # Map the pixel range to layer counts
            if pixel == 255:
                return 0

            # For non-white pixels, map to 1..distinct_colors layers
            # Darker pixels get more layers
            layer_interval = 255 / self.distinct_colors

            # Invert: darker (lower values) = more layers
            total_layers = (self.distinct_colors - int(pixel / layer_interval))

            # Ensure at least 1 layer for non-white pixels
            return max(1, total_layers)

        # Vectorize the function for efficient array processing
        pixel_to_height_vec = np.vectorize(pixel_to_layer_count)

        return pixel_to_height_vec(image_arr)

    def add_transparent_layers(self, transparent_layers=1, color_pixels_only=False):
        """
        Add transparent layers to the heightmap.

        Args:
            transparent_layers: Number of transparent layers to add (default: 5)
            color_pixels_only: If True, only add transparent layers to non-zero pixels (default: False)

        Returns:
            self (for method chaining)
        """

        def pixel_to_layer_count(pixel):
            if pixel != 0 or not color_pixels_only:
                return pixel + transparent_layers
            else:
                return pixel

        # Vectorize the function for efficient array processing
        pixel_to_height_vec = np.vectorize(pixel_to_layer_count)

        # Update the heightmap
        self.heightmap = pixel_to_height_vec(self.heightmap)

        return self

    def add_border(self, border_thickness=10, border_value=5):
        """
        Add a border of specified height around the heightmap.
        This uses dilation to expand the logo and adds a border around it.

        Args:
            border_thickness: Width of the border in pixels (default: 10)
            border_value: Height value for the border pixels (default: 5, typically transparent_layers)

        Returns:
            self (for method chaining)
        """
        result = self.heightmap.copy()

        # 0 means no height, so all non-0 values are a pixel
        mask = self.heightmap > 0

        # Manual dilation using sliding window
        padded = np.pad(mask, border_thickness, mode='constant', constant_values=False)
        dilated = np.zeros_like(mask)

        for i in range(-border_thickness, border_thickness + 1):
            for j in range(-border_thickness, border_thickness + 1):
                dilated |= padded[border_thickness + i: border_thickness + i + mask.shape[0],
                           border_thickness + j: border_thickness + j + mask.shape[1]]

        # Border is dilated minus original
        border_mask = dilated & ~mask
        result[border_mask] = border_value

        self.heightmap = result
        return self

    def smooth(self, sigma=1.0, edge_width=3):
        """
        Smooth only the outer edges of the heightmap to reduce jagged edges.
        Uses Gaussian blur only on pixels near the boundary.

        Args:
            sigma: Standard deviation for Gaussian kernel (default: 1.0)
                   Higher values = more smoothing
            edge_width: Width of edge region to smooth in pixels (default: 3)

        Returns:
            self (for method chaining)
        """
        from scipy.ndimage import gaussian_filter, binary_erosion

        # Create mask of non-zero pixels
        mask = self.heightmap > 0

        # Erode the mask to find interior pixels
        # Pixels that remain after erosion are interior (not edges)
        struct = np.ones((edge_width * 2 + 1, edge_width * 2 + 1))
        interior = binary_erosion(mask, structure=struct)

        # Edge pixels are non-zero pixels that are NOT interior
        edge_mask = mask & ~interior

        # Apply Gaussian smoothing to entire heightmap
        smoothed = gaussian_filter(self.heightmap, sigma=sigma)

        # Only apply smoothing to edge pixels
        # Interior pixels keep original values, edges get smoothed
        self.heightmap = np.where(edge_mask, smoothed, self.heightmap)

        return self

    def scale_to_layer_height(self, layer_height=0.1):
        """
        Scale the heightmap values by the layer height.
        Converts layer counts to actual physical heights in mm as floats.

        Args:
            layer_height: Height of each layer in mm (default: 0.1)

        Returns:
            self (for method chaining)
        """
        self.heightmap = self.heightmap.astype(np.float64) * layer_height
        return self

    def to_stl(self, output_file='output.stl', scale_xy=1.0, scale_z=1.0):
        """
        Convert a 2D array of heights into an STL file.

        Parameters:
        - height_array: 2D numpy array where each value is the height
        - output_file: name of the output STL file
        - scale_xy: scaling factor for x and y dimensions
        - scale_z: scaling factor for z (height) dimension
        """
        height_array = self.heightmap
        rows, cols = height_array.shape

        # Create vertices for the top surface
        vertices = []
        faces = []

        # Generate vertices for each point in the grid
        for i in range(rows):
            for j in range(cols):
                x = j * scale_xy
                y = i * scale_xy
                z = height_array[i, j] * scale_z
                vertices.append([x, y, z])

        # Generate triangular faces for the top surface
        # Each grid cell creates 2 triangles
        # Skip cells where all corners have zero height
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get heights of the 4 corners
                h1 = height_array[i, j]
                h2 = height_array[i, j + 1]
                h3 = height_array[i + 1, j]
                h4 = height_array[i + 1, j + 1]

                # Skip if all corners are at zero height
                if max(h1, h2, h3, h4) < 0.001:
                    continue

                # Indices of the 4 corners of current cell
                v1 = i * cols + j
                v2 = i * cols + (j + 1)
                v3 = (i + 1) * cols + j
                v4 = (i + 1) * cols + (j + 1)

                # Two triangles per cell
                faces.append([v1, v2, v4])
                faces.append([v1, v4, v3])

        # Add bottom face (flat at z=0)
        for i in range(rows):
            for j in range(cols):
                x = j * scale_xy
                y = i * scale_xy
                vertices.append([x, y, 0])

        # Bottom faces
        # Skip cells where all corners have zero height
        offset = rows * cols
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get heights of the 4 corners
                h1 = height_array[i, j]
                h2 = height_array[i, j + 1]
                h3 = height_array[i + 1, j]
                h4 = height_array[i + 1, j + 1]

                # Skip if all corners are at zero height
                if max(h1, h2, h3, h4) < 0.001:
                    continue

                v1 = offset + i * cols + j
                v2 = offset + i * cols + (j + 1)
                v3 = offset + (i + 1) * cols + j
                v4 = offset + (i + 1) * cols + (j + 1)

                # Reverse winding order for bottom
                faces.append([v1, v4, v2])
                faces.append([v1, v3, v4])

        # Add side walls
        # Only add walls where at least one vertex has non-zero height
        # Left edge
        for i in range(rows - 1):
            h1 = height_array[i, 0]
            h2 = height_array[i + 1, 0]
            if max(h1, h2) < 0.001:
                continue

            top1 = i * cols
            top2 = (i + 1) * cols
            bot1 = offset + i * cols
            bot2 = offset + (i + 1) * cols
            faces.append([top1, bot1, top2])
            faces.append([top2, bot1, bot2])

        # Right edge
        for i in range(rows - 1):
            h1 = height_array[i, cols - 1]
            h2 = height_array[i + 1, cols - 1]
            if max(h1, h2) < 0.001:
                continue

            top1 = i * cols + (cols - 1)
            top2 = (i + 1) * cols + (cols - 1)
            bot1 = offset + i * cols + (cols - 1)
            bot2 = offset + (i + 1) * cols + (cols - 1)
            faces.append([top1, top2, bot1])
            faces.append([top2, bot2, bot1])

        # Front edge
        for j in range(cols - 1):
            h1 = height_array[0, j]
            h2 = height_array[0, j + 1]
            if max(h1, h2) < 0.001:
                continue

            top1 = j
            top2 = j + 1
            bot1 = offset + j
            bot2 = offset + j + 1
            faces.append([top1, top2, bot1])
            faces.append([top2, bot2, bot1])

        # Back edge
        for j in range(cols - 1):
            h1 = height_array[rows - 1, j]
            h2 = height_array[rows - 1, j + 1]
            if max(h1, h2) < 0.001:
                continue

            top1 = (rows - 1) * cols + j
            top2 = (rows - 1) * cols + j + 1
            bot1 = offset + (rows - 1) * cols + j
            bot2 = offset + (rows - 1) * cols + j + 1
            faces.append([top1, bot1, top2])
            faces.append([top2, bot1, bot2])

        # Create the mesh
        vertices = np.array(vertices)
        faces = np.array(faces)

        # Create mesh object
        surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                surface_mesh.vectors[i][j] = vertices[face[j]]

        # Write to file
        surface_mesh.save(output_file)
        print(f"STL file saved as {output_file}")


def scale_stl(file, scale=0.2):
    """
    Scale the STL file with a given factor.
    """
    model = openstl.read(file)  # np.array of shape (N,4,3) -> normal, v0,v1,v2
    model[:, 1:4, 0] *= scale  # Scale X coordinates
    model[:, 1:4, 1] *= scale  # Scale Y coordinates
    openstl.write(file, model, openstl.format.binary)

openstl.set_activate_overflow_safety(False)

# these parameters are for the elixir logo
# process_factor = 1.5 * 1
# stl_scaledown = 0.19 / 1
# layer_height=0.1
# transparent_layers = False
# colors = 20
# silhouette = True
# fill_holes = True

# the erlang logo doesn't need scaling since it's much simpler.
process_factor = 1
stl_scaledown = 0.1
layer_height=0.1
transparent_layers = True
colors = 2
silhouette =  False
fill_holes = False


# Read in the image and do some processing.
# This involves removing the background alpha, quantizing, resizing, etc.
image = ImageContainer.from_image("logos/erlang.png")
image = image.process_logo(scale=process_factor, colors=colors, median_size=13, border_width=15, fill_holes=fill_holes)

# Create the stl for the colored layers of the logo.
# This has to be put onto a silhouette model. Generated below.
heightmap = Heightmap(image)
heightmap = heightmap.scale_to_layer_height(layer_height=layer_height)

# for the erlang logo we can add the transparent layer immediately.
if transparent_layers:
    heightmap = heightmap.add_transparent_layers(color_pixels_only=False)

heightmap.to_stl("output/erlang.stl")
scale_stl("output/erlang.stl", scale=stl_scaledown)

if silhouette:
    # Create the silhouette of the logo to put the colored layer on top of.
    image = image.create_silhouette(border_width=0).median_filter(size=13).create_silhouette(border_width=10)
    heightmap = Heightmap(image)
    heightmap = heightmap.scale_to_layer_height(layer_height=layer_height)
    heightmap.to_stl("output/erlang_silhouette.stl")
    scale_stl("output/erlang_silhouette.stl", scale=stl_scaledown)