from PIL import Image
import os

# Paths to the folders for color images, black-and-white images, and output
color_image_folder = "/images/SCC"
bw_image_folder = "/images/predict"
output_folder = "/images/segment"

def overlay_images(color_img_path, bw_img_path, output_path):
    """
    Overlay a black-and-white image onto a color image, highlighting white areas in red.

    Args:
        color_img_path (str): Path to the color image.
        bw_img_path (str): Path to the black-and-white image.
        output_path (str): Path to save the overlaid image.
    """
    # Open the color and black-and-white images
    color_img = Image.open(color_img_path)
    bw_img = Image.open(bw_img_path)
    
    # Resize the black-and-white image to match the size of the color image
    bw_img = bw_img.resize(color_img.size, Image.LANCZOS)
    
    # Change white areas in the black-and-white image to transparent red
    bw_img_data = bw_img.getdata()
    overlay_data = []
    for pixel in bw_img_data:
        if 128 <= pixel[0] <= 255:  # Convert pixels in the range of 128 to 255 to transparent
            overlay_data.append((0, 0, 0, 0))
        else:
            overlay_data.append((255, 255, 255, 255))  # Keep other pixels white

    # Create an RGBA overlay image
    overlay_img = Image.new("RGBA", color_img.size)
    overlay_img.putdata(overlay_data)
    
    # Composite the color image with the overlay image
    result_img = Image.alpha_composite(color_img.convert("RGBA"), overlay_img)
    
    # Save the resulting image to the specified output path
    result_img.save(output_path, format="PNG")

# Process each image pair in the color and black-and-white folders
for filename in os.listdir(color_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust for supported file extensions
        color_img_path = os.path.join(color_image_folder, filename)
        bw_img_path = os.path.join(bw_image_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        overlay_images(color_img_path, bw_img_path, output_path)
