import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from detection import hough_circle_transform

def read_annotations(txt_file, scale_factor):
    """
    Read coin annotations from a text file and scale them down.
    """
    annotations = []
    labels = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        num_coins = int(lines[0].strip())
        for line in lines[1:]:
            x, y, coin_type, radius = line.strip().split()
            x = int(float(x) / scale_factor)
            y = int(float(y) / scale_factor)
            radius = int(float(radius) / scale_factor)
            annotations.append((x, y, radius))
            coin_type = int(coin_type)
            # Label as 1 if penny (coin_type == 1), else 0
            if coin_type == 1:
                label = 1  # Penny
            else:
                label = 0  # Not a penny
            labels.append(label)
    return annotations, labels

def match_circles_to_annotations(detected_circles, annotations):
    """
    Match detected circles to annotations based on proximity.
    """
    matched_indices = []
    used_circles = set()
    used_annotations = set()
    for i, (x_ann, y_ann, r_ann) in enumerate(annotations):
        min_dist = float('inf')
        matched_circle_idx = -1
        for j, (x_det, y_det, r_det) in enumerate(detected_circles):
            if j in used_circles:
                continue
            dist = np.sqrt((x_ann - x_det) ** 2 + (y_ann - y_det) ** 2)
            if dist < min_dist and dist < r_ann:  # Adjust distance threshold as needed
                min_dist = dist
                matched_circle_idx = j
        if matched_circle_idx != -1:
            matched_indices.append((matched_circle_idx, i))
            used_circles.add(matched_circle_idx)
            used_annotations.add(i)
    return matched_indices

def extract_coin_features(image, circle):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
    coin_region = cv2.bitwise_and(image, image, mask=mask)
    hsv_coin = cv2.cvtColor(coin_region, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_coin[mask == 255]
    if hsv_values.size == 0:
        return None
    avg_hsv = np.mean(hsv_values, axis=0)
    return avg_hsv

def main():
    # Directory containing training images and annotations
    dir = 'test_cases'  
    X = []
    Y = []

    scale_factor = 5  # The downsampling factor

    for filename in os.listdir(dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_file = os.path.join(dir, filename)
            annotation_file = os.path.join(dir, filename.replace('.png', '.txt').replace('.jpg', '.txt'))

            image = cv2.imread(image_file)
            if image is None:
                print(f"Warning: Could not read image {image_file}")
                continue

            # Downsample image
            height, width = image.shape[:2]
            downsampled_image = cv2.resize(
                image, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_LINEAR)

            # Read annotations (labels)
            annotations, labels = read_annotations(annotation_file, scale_factor)

            # Normalize image
            min_value = np.min(downsampled_image)
            max_value = np.max(downsampled_image)
            normalized_image = (((downsampled_image - min_value) / (max_value - min_value)) * 255).astype(np.uint8)

            # Detect circles using manual Hough transform
            min_radius = 18
            max_radius = 27
            threshold = 0.65  # Adjust as needed
            detected_circles = hough_circle_transform(normalized_image, min_radius, max_radius, threshold)

            # Match detected circles to annotations
            matched_indices = match_circles_to_annotations(detected_circles, annotations)

            if len(matched_indices) == 0:
                print(f"No matches found in {filename}")
                continue

            for (circle_idx, ann_idx) in matched_indices:
                circle = detected_circles[circle_idx]
                label = labels[ann_idx]
                features = extract_coin_features(downsampled_image, circle)
                if features is not None:
                    X.append(features)
                    Y.append(label)

    if not X:
        print("No training data collected.")
        return

    X = np.array(X)
    Y = np.array(Y)

    # Feature normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save the scaler for future use
    joblib.dump(scaler, 'penny_scaler.pkl')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Train SVM classifier
    svm = SVC(kernel='rbf', gamma='scale')
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.2f}")

    # Save the trained model
    joblib.dump(svm, 'penny_svm_model.pkl')
    print("Model saved as 'penny_svm_model.pkl'")

if __name__ == '__main__':
    main()
