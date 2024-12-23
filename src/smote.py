# smote.py

import cv2
import numpy as np
import os
import random


def translate_image(image, annotations, max_shift=15):
    """
    Perform Translations on the images 
    """
    h, w = image.shape[:2]
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (w, h))
    translated_annotations = [(x + tx, y + ty, coin_type, radius) for x, y, coin_type, radius in annotations]
    return translated_image, np.array(translated_annotations)


def rotate_image(image, annotations, max_angle=15):
    """
    Perform Rotations on the image
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_annotations = [(M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2], coin_type, radius) for x, y, coin_type, radius in annotations]
    return rotated_image, np.array(rotated_annotations)



def flip_image(image, annotations, flip_code=1):
    """ Flip image across an axis """
    flipped_image = cv2.flip(image, flip_code)
    h, w = image.shape[:2]
    flipped_annotations = []
    for x, y, coin_type, radius in annotations:
        if flip_code == 1:
            flipped_annotations.append((w - x, y, coin_type, radius))
        elif flip_code == 0:
            flipped_annotations.append((x, h - y, coin_type, radius))
        else:
            flipped_annotations.append((w - x, h - y, coin_type, radius))
    return flipped_image, np.array(flipped_annotations)


def augment_data(image, annotations, target_samples=15):
    """Augment the data for the number of samples"""
    augmented_images = []
    augmented_annotations = []
    
    while len(augmented_images) < target_samples:
        #
        augmented_image, augmented_annots = image, annotations
        # Ransomly pick a translation
        choice = random.choice(['translate', 'rotate', 'flip'])
        # Perform transformations based on choice
        if choice == 'translate':
            augmented_image, augmented_annots = translate_image(augmented_image, augmented_annots)
        elif choice == 'rotate':
            augmented_image, augmented_annots = rotate_image(augmented_image, augmented_annots)
        elif choice == 'flip':
            flip_code = random.choice([0, 1, -1])
            augmented_image, augmented_annots = flip_image(augmented_image, augmented_annots, flip_code)
        
        augmented_images.append(augmented_image)
        augmented_annotations.append(augmented_annots)
    
    return augmented_images, augmented_annotations


def save_augmented_data(augmented_images, augmented_annotations, base_filename, output_dir='augmented_data'):
    os.makedirs(output_dir, exist_ok=True) # Make directory if it doesnt exit
    # Creare the augmented files and annotations
    for i, (aug_img, aug_annot) in enumerate(zip(augmented_images, augmented_annotations)):
        img_filename = os.path.join(output_dir, f'{base_filename}_augmented_{i}.png')
        cv2.imwrite(img_filename, aug_img)
        
        annot_filename = os.path.join(output_dir, f'{base_filename}_augmented_{i}.txt')
        with open(annot_filename, 'w') as f:
            f.write(f"{len(aug_annot)}\n")
            for (x, y, coin_type, radius) in aug_annot:
                f.write(f"{int(x)} {int(y)} {int(coin_type)} {int(radius)}\n")


def read_annotations(txt_file, scale_factor=1):
    # Reed the annotations file and extract the coin information
    annotations = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            x, y, coin_type, radius = line.strip().split()
            annotations.append((float(x) / scale_factor, float(y) / scale_factor, int(coin_type), float(radius) / scale_factor))
    return np.array(annotations)


def augment_directory(input_dir, output_dir='augmented_data', target_samples=15, scale_factor=1):
    """ 
    Perform augmentations and save to a target directory
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            annotation_path = os.path.join(input_dir, filename.replace('.png', '.txt'))

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            annotations = read_annotations(annotation_path, scale_factor)
            
            # Generate augmented samples
            augmented_images, augmented_annotations = augment_data(image, annotations, target_samples=target_samples)

            # Save the augmented images and annotations
            base_filename = os.path.splitext(filename)[0]
            save_augmented_data(augmented_images, augmented_annotations, base_filename, output_dir)


if __name__ == "__main__":
    input_dir = 'test_cases'  # Set the input directory with .png images and .txt annotations
    output_dir = 'augmented_data'
    target_samples =3  # Set the number of augmented samples per image

    augment_directory(input_dir, output_dir, target_samples=target_samples)
