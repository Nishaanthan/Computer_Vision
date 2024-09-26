import sys
sys.path.append(".")
import os
import cv2
from utils import show_images, save_images, scale_down, separate_channels

if __name__ == "__main__":
    ## TODO 3.1
    ## Load Image
    ## Show it on screen
    ## Note: implement show_images in utils/functions.py
    input_path = os.path.join("resources", "img.png")
    output_path = os.path.join(os.getcwd(), "results")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    show_images([img], ["Display Image"])

    ## TODO 3.2
    ## Resize Image by a factor of 0.5
    ## Show it on screen
    ## Save as small.jpg
    ## Note: implement save_images, scale_down in utils/functions.py
    small_img = scale_down(img)
    show_images([small_img], ["Small Image"])
    save_images([small_img], [os.path.join(output_path, "small.jpg")])

    ## TODO 3.3
    ## Create and save 3 single-channel images from small image
    ## one image each channel (r, g, b)
    ## Display the channel-images on screen
    ## Note: implement separate_channels in utils/functions.py
    blue, green, red = separate_channels(img)
    show_images([img, blue, green, red], ["Original", "Blue", "Green", "Red"])
    save_images([blue, green, red], [os.path.join(output_path, f"{name}.png") for name in ["blue", "green", "red"]])
