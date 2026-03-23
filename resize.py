import cv2
import os

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output_images"
TARGET_SIZE = 216


def crop_center_square(img):
    h, w = img.shape[:2]
    side = min(h, w)

    x = (w - side) // 2
    y = (h - side) // 2

    return img[y:y + side, x:x + side]


def process_images():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, filename)

        if not os.path.isfile(input_path):
            continue

        img = cv2.imread(input_path)
        if img is None:
            print(f"Skipped: {filename} (not an image or cannot read)")
            continue

        img = crop_center_square(img)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, img)

        print(f"Processed: {filename}")

    print("Done.")


if __name__ == "__main__":
    process_images()