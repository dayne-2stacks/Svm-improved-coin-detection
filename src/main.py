import cv2
import numpy as np
import joblib
import sys
from detection import hough_circle_transform, extract_coin_features

# Load the trained SVM model and scaler
# My SVM model was u=obtained by augmenting the original samples provided in project one by conducting translations on the dataset
# These include rotations and translations as seen in smote.py
# The model was simply created by performing the hough transform and matching the annotations file with the manual hough transform circles based
#  on Euclidean distance and then add to a SVM model. The SVM model is tested with 20% of the dataset and trained with 80%.
penny_svm_model = joblib.load('penny_svm_model.pkl')
# Helper since my samples were downsampled
# Helps normalize the model since I downsampled the images.
scaler = joblib.load('penny_scaler.pkl')

def classify_coin(image, circle):
    radius = circle[2]
    # Extract features for the coin
    features = extract_coin_features(image, circle)
    if features is None:
        return "Unknown"
    # Normalize features
    features = scaler.transform([features])
    # Using Machine Learing here to detect if penny or not
    # Predict using the SVM model 
    is_penny = penny_svm_model.predict(features)[0]
    if is_penny == 1:
        return 1  # Penny
    else:
        # If not a penny, classify based on radius
        if radius in range(20, 23):  # Nickel radius range
            return 5
        elif radius in range(18, 20):  # Dime radius range
            return 10
        elif radius in range(23, 27):  # Quarter radius range
            return 25
        else:
            return "Unknown"

def main():
    # Read the image file name from standard input
    image_file = input().strip()

    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not read image {image_file}")
        sys.exit(1)

    # Downsample by a factor of 5
    height, width = image.shape[:2]
    new_width = width // 5
    new_height = height // 5

    # Resize the image
    downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Define minimum and maximum radius for downsampled images and a voting threshold
    min_radius = 18
    max_radius = 27
    threshold = 0.64

    # Find the minimum and maximum pixel values
    min_value = np.min(downsampled_image)
    max_value = np.max(downsampled_image)

    # Apply normalization to increase contrast
    normalized_image = (((downsampled_image - min_value) /
                        (max_value - min_value)) * 255).astype(np.uint8)

    # Detect circles
    circles = hough_circle_transform(
        normalized_image, min_radius, max_radius, threshold)

    # Output the results with classification
    print(f"{len(circles)}")
    for circle in circles:
        coin_type = classify_coin(downsampled_image, circle)
        x_coord = circle[0] * 5  # Adjust back to original image size
        y_coord = circle[1] * 5
        print(f"{x_coord} {y_coord} {coin_type}")


if __name__ == "__main__":
    main()
