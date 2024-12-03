import requests
import cv2
import numpy as np
from rembg import remove
import os

class ImageCompositor:
    def __init__(self, deepai_api_key):
        self.deepai_api_key = deepai_api_key  # DeepAI API key for generating animation
    
    def remove_background(self, image):
        """
        Remove background from the input image.
        """
        try:
            _, encoded_image = cv2.imencode('.png', image)
            binary_image = encoded_image.tobytes()

            output = remove(binary_image)
            output_image = np.frombuffer(output, dtype=np.uint8)
            output_image = cv2.imdecode(output_image, cv2.IMREAD_UNCHANGED)

            if output_image.shape[2] == 4:  # RGBA to BGR
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGR)

            return output_image
        except Exception as e:
            print(f"Error during background removal: {e}")
            return image  # Fallback to original image if error occurs

    def generate_animation(self, image_path):
        """
        Generate animation (e.g., adding movement or effects) using DeepAI.
        """
        # Upload image to DeepAI API for animation
        try:
            response = requests.post(
                "https://api.deepai.org/api/text2img",  # You can use other DeepAI APIs based on your use case
                files={'image': open(image_path, 'rb')},
                headers={'api-key': self.deepai_api_key}
            )
            if response.status_code == 200:
                result = response.json()
                return result['output_url']  # This is the URL to the generated animation or result
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"Error during animation generation: {e}")
            return None
    
    def composite_images(self, image1_path, image2_path, output_path='composite_video.mp4'):
        """
        Composite two images together and generate animation with effects.
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

        # Generate animation for the first image
        animation_url_1 = self.generate_animation(image1_path)
        animation_url_2 = self.generate_animation(image2_path)

        if animation_url_1 and animation_url_2:
            print(f"Generated animation for image1: {animation_url_1}")
            print(f"Generated animation for image2: {animation_url_2}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (canvas.shape[1], canvas.shape[0]))

        # Create a short animation (3 seconds)
        for _ in range(90):  # 3 seconds at 30 fps
            out.write(canvas)  # Add your generated animation frame here
        
        out.release()
        print(f"Composite video saved to {output_path}")

def main():
    deepai_api_key = 'YOUR_DEEPAI_API_KEY'  # Replace with your DeepAI API key
    compositor = ImageCompositor(deepai_api_key)

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
