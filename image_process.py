from PIL import Image

def load_image_and_get_rgb(image_path):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert the image to RGB mode if it's not already
        img_rgb = img.convert("RGB")

        # Get the width and height of the image
        width, height = img_rgb.size

        # Initialize an empty list to store RGB values
        rgb_values = []

        # Loop through each pixel in the image
        for y in range(height):
            for x in range(width):
                r, g, b = img_rgb.getpixel((x, y))
                rgb_values.append((r, g, b))

        return rgb_values
    except Exception as e:
        print("An error occurred:", e)
        return None

# Replace 'image_path' with the actual path of your image
image_path = "/Users/ericchan/Desktop/thales.jpeg"
rgb_values = load_image_and_get_rgb(image_path)
if rgb_values:
    print("RGB values of the image:", rgb_values)