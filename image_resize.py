from PIL import Image
import os


def resize_images(input_folder, output_folder):
    """
    Resize all .png images in the specified folder to have a maximum side length of 512 pixels
    and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where resized images will be saved.
    """

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all files in the input folder
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Skip files that are not .png
        if not file_name.lower().endswith('.png'):
            continue

        # Create the path to the image file
        input_path = os.path.join(input_folder, file_name)

        # Open the image
        image = Image.open(input_path)

        # Get the image dimensions
        width, height = image.size

        # Resize the image so that the longest side is 512 pixels
        if width > height:
            new_width = 512
            new_height = int(height * (512 / width))
        else:
            new_width = int(width * (512 / height))
            new_height = 512

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create the path for the resized image
        output_path = os.path.join(output_folder, file_name)

        # Save the resized image
        resized_image.save(output_path)
        print(f"Resized and saved: {output_path}")


if __name__ == "__main__":
    # Specify the path to the input folder
    input_folder = 'images/classification'
    # Specify the path to the output folder
    output_folder = 'images/classification'

    resize_images(input_folder, output_folder)
