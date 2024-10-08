import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np
import random

# Function to visualize images side by side
def visualize(images, titles, suptitle=None, cmap=None):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap=cmap if i == 1 else 'gray')  # Show corners in color, grayscale in original
        plt.title(titles[i])
        plt.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()

def detect_and_crop_license_plate(binary_image, original_image):
    """
    Detects and crops the license plate from the given images.

    Parameters:
    - binary_image: Preprocessed binary image.
    - original_image: The original input image.

    Returns:
    - license_plate: Cropped image of the detected license plate or None if not found.
    - img_with_contour: Image of the car with the detected license plate contour drawn or None if not found.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area in descending order and take top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate_contour = None

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        # Check if the approximated contour has 4 points (quadrilateral)
        if len(approx) == 4:
            # Compute the bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Define a range for aspect ratio typical of license plates
            if 2 <= aspect_ratio <= 6:
                # Further check for parallelism of edges
                angles = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % 4][0]
                    pt3 = approx[(i + 2) % 4][0]

                    v1 = pt2 - pt1
                    v2 = pt3 - pt2

                    angle = angle_between(v1, v2)
                    angles.append(angle)

                if all(80 <= angle <= 100 for angle in angles):
                    license_plate_contour = approx
                    break  # Stop once the likely license plate is found

    if license_plate_contour is not None:
        # Draw the detected contour on the image (for visualization)
        img_copy = original_image.copy()
        cv2.drawContours(img_copy, [license_plate_contour], -1, (0, 255, 0), 3)

        # Obtain a top-down view of the license plate using perspective transform
        license_plate = four_point_transform(original_image, license_plate_contour.reshape(4, 2))

        # Display the results
        # Use cv2_imshow if in Jupyter; otherwise, use cv2.imshow
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(license_plate)
            cv2_imshow(img_copy)
        except ImportError:
            cv2.imshow("Cropped License Plate", license_plate)
            cv2.imshow("Detected License Plate", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("License plate detected")
        # Return both the cropped license plate and the image with contour
        return license_plate, img_copy

    else:
        print("License plate not found.")
        return None, None

def angle_between(v1, v2):
    """
    Calculates the angle between two vectors in degrees.
    """
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    unit_v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else v1
    unit_v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) != 0 else v2
    dot_product = np.dot(unit_v1, unit_v2)
    # Clamp the dot_product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find corners
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to obtain a top-down view of the detected license plate.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def OCR_image(license_plate):
    if license_plate is None or license_plate.size == 0:
        print("Error: The license plate image is empty or not loaded correctly.")
        return None

    upscale_factor = 8
    upscaled_license_plate = cv2.resize(license_plate, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    license_plate_2 = upscaled_license_plate.copy()


    gray = cv2.cvtColor(upscaled_license_plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                #  cv2.THRESH_BINARY_INV, 11, 2)

    #dilated operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    visualize([dilated],
                ["Dilated"], cmap='gray')