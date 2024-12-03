import cv2
import numpy as np
from rembg import remove
import mediapipe as mp
import os

class ImageCompositor:
    def __init__(self):
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def remove_background(self, image):
        """
        Remove background from the input image.
        """
        try:
            # Convert image to binary buffer
            _, encoded_image = cv2.imencode('.png', image)
            binary_image = encoded_image.tobytes()

            # Use rembg to remove the background
            output = remove(binary_image)

            # Convert the output binary buffer back to OpenCV format
            output_image = np.frombuffer(output, dtype=np.uint8)
            output_image = cv2.imdecode(output_image, cv2.IMREAD_UNCHANGED)

            # Ensure the output image is in BGR format (remove alpha if present)
            if output_image.shape[2] == 4:  # RGBA to BGR
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGR)

            return output_image
        except Exception as e:
            print(f"Error during background removal: {e}")
            return image  # Fallback to the original image if error occurs

    def detect_pose(self, image):
        """
        Detect pose landmarks in the image.
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find poses
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True
            ) as pose:
                results = pose.process(rgb_image)

            return results
        except Exception as e:
            print(f"Error during pose detection: {e}")
            return None

    def composite_images(self, image1_path, image2_path, output_path='composite_video.mp4'):
        """
        Composite two images together in a short animated video.
        """
        # Read input images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images.")

        # Remove backgrounds
        img1_nobg = self.remove_background(img1)
        img2_nobg = self.remove_background(img2)

        # Create a blank canvas (white background)
        canvas_height = max(img1_nobg.shape[0], img2_nobg.shape[0])
        canvas_width = img1_nobg.shape[1] + img2_nobg.shape[1]
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Place images side by side
        canvas[:img1_nobg.shape[0], :img1_nobg.shape[1]] = img1_nobg
        canvas[:img2_nobg.shape[0], img1_nobg.shape[1]:] = img2_nobg

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (canvas.shape[1], canvas.shape[0]))

        # Create a short animation (3 seconds)
        for frame_idx in range(90):  # 3 seconds at 30 fps
            # Add fade-in effect
            alpha = frame_idx / 90.0
            overlay = (canvas * alpha).astype(np.uint8)
            out.write(overlay)

        out.release()
        print(f"Composite video saved to {output_path}")

def main():
    compositor = ImageCompositor()

    # Prompt user for image paths
    image1_path = input("Enter the path to the first image: ").strip()
    image2_path = input("Enter the path to the second image: ").strip()

    # Check if files exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("One or both image paths are invalid.")
        return

    # Prompt user for output path
    output_path = input("Enter the output video path (press Enter for default 'composite_video.mp4'): ").strip()
    if not output_path:
        output_path = 'composite_video.mp4'

    try:
        compositor.composite_images(image1_path, image2_path, output_path)
        print(f"Video created successfully: {output_path}")
    except Exception as e:
        print(f"An error occurred during video creation: {e}")

# Ensure the script can be run directly or imported
if __name__ == "__main__":
    main()
