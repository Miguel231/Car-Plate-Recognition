import cv2
import numpy as np

class PlateFinder:
    def __init__(self, min_plate_area=100, max_plate_area=15000):
        self.min_plate_area = min_plate_area
        self.max_plate_area = max_plate_area

    def preprocess(self, image):
        """ Convert image to grayscale and apply Laplacian edge detection """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_8U)
        _, thresh = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def find_rectangular_contours(self, thresh):
        """ Find and filter rectangular contours """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_contours = []

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is a rectangle or square
            if len(approx) == 4:  # Rectangle or square
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(contour)
                aspect_ratio = float(w) / h if h != 0 else 0

                if self.min_plate_area <= area <= self.max_plate_area and 0.5 <= aspect_ratio <= 2:
                    rectangular_contours.append(approx)
        
        return rectangular_contours

    def extract_plate(self, image, contour):
        """ Extract and validate the plate from a rectangular contour """
        x, y, w, h = cv2.boundingRect(contour)
        plate_img = image[y:y + h, x:x + w]

        # Convert plate image to grayscale and apply adaptive thresholding
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check if the plate has a valid character pattern
        return thresh_plate

    def find_possible_plates(self, image):
        """ Detect possible plates in the image """
        thresh = self.preprocess(image)
        rectangular_contours = self.find_rectangular_contours(thresh)

        plates = []
        for contour in rectangular_contours:
            plate_img = self.extract_plate(image, contour)
            plates.append(plate_img)
        
        return plates

    def check_plate(self, plate_img, plate_text):
        """ Check if the given text matches the extracted plate """
        # Convert plate image to grayscale and apply thresholding
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract text from the plate using OCR
        import pytesseract
        extracted_text = pytesseract.image_to_string(thresh_plate, config='--psm 8').strip()
        print('TEXT:', extracted_text)

        # Compare the extracted text with the provided plate text
        return extracted_text == plate_text

if __name__ == "__main__":
    plate_finder = PlateFinder(min_plate_area=5000, max_plate_area=15000)
    image_path = 'C:/Users/larar/OneDrive/Documentos/Escritorio/car.jpeg'
    plate_text_to_check = '1712JPF'  # Plate text to verify

    image = cv2.imread(image_path)
    possible_plates = plate_finder.find_possible_plates(image)
    print('PLATES',possible_plates)

    for plate_img in possible_plates:
        if plate_finder.check_plate(plate_img, plate_text_to_check):
            print("Plate found!")
            break
    else:
        print("Plate not found.")
