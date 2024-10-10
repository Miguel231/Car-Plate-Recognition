import cv2
from License_Plate_Detection.first_collab import visualize

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
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    min_height = 10
    min_width = 5
    max_aspect_ratio = 2

    characters = []
    for contour in sorted_contours:
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)

        x1, y1, w1, h1 = cv2.boundingRect(contour)

        if h > min_height and w > min_width and w / h <= max_aspect_ratio:
            char = dilated[y:y+h, x:x+w]
            characters.append(char)

            cv2.rectangle(upscaled_license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    visualize([hull],
                ["Hull"], cmap='gray')

    visualize([upscaled_license_plate],
                ["Segmented Characters"], cmap='gray')
    return characters



