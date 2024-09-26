import numpy as np
import os
import cv2
import sys
sys.path.append(".")
from utils import show_images, compute_harris_response, detect_corners, save_images, detect_edges, draw_mask, draw_points


if __name__ == "__main__":
    input_path = os.path.join("resources", "input.jpg")
    output_path = os.path.join(os.getcwd(), "results")
    input = cv2.imread(input_path, cv2.IMREAD_COLOR)

    gray_float_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    R, A, B, C, Idx, Idy = compute_harris_response(gray_float_image, k=.06)
    image_names = ["R", "A", "B", "C", "Idx", "Idy"]
    # 0 Centered images are shifted from black to gray for 0. by adding .5
    show_images([input, .5 + R, A, B, .5 + C, .5 + Idx, .5 + Idy], ["input"] + image_names, tile_yx=(400, 400))
    output_paths = [os.path.join(output_path, f"{name}.png") for name in image_names]
    save_images([.5 + R, A, B, .5 + C, .5 + Idx, .5 + Idy], output_paths)

    points = detect_corners(R, threshold=.1)
    drawn_points = draw_points(input, points, color=(0, 255, 0))
    show_images([input, .5 + R, drawn_points], ["Input", "Harris Response", "Key points"], tile_yx=(400, 400))
    save_images([drawn_points], [os.path.join(output_path, "points.png")])

    edges = detect_edges(R, edge_threshold=-.01)
    drawn_edges = draw_mask(input, edges, color=(255, 0, 0))
    show_images([.5 + R, edges, drawn_edges], ["Harris Response", "Edges", "Drawn Edges"], tile_yx=(400, 400))
    save_images([edges, drawn_edges], [os.path.join(output_path, f) for f in ["edges.png", "drawn_edges.png"]])

