import streamlit as st
import cv2
import numpy as np
from rembg import remove
import os
from PIL import Image
import tempfile
import webbrowser
from io import BytesIO

class ImageCompositor:
    def remove_background(self, image):
        try:
            _, encoded_image = cv2.imencode('.png', image)
            binary_image = encoded_image.tobytes()
            output = remove(binary_image)
            output_image = np.frombuffer(output, dtype=np.uint8)
            output_image = cv2.imdecode(output_image, cv2.IMREAD_UNCHANGED)

            if output_image.shape[2] == 4:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2BGR)
            return output_image
        except Exception as e:
            st.error(f"Error during background removal: {e}")
            return image

    def composite_images(self, img1, img2):
        try:
            img1_nobg = self.remove_background(img1)
            img2_nobg = self.remove_background(img2)

            canvas_height = max(img1_nobg.shape[0], img2_nobg.shape[0])
            canvas_width = img1_nobg.shape[1] + img2_nobg.shape[1]
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

            canvas[:img1_nobg.shape[0], :img1_nobg.shape[1]] = img1_nobg
            canvas[:img2_nobg.shape[0], img1_nobg.shape[1]:] = img2_nobg
            return canvas
        except Exception as e:
            st.error(f"Error during image compositing: {e}")
            return None

# Streamlit UI
st.title("Image Compositor Web App")

st.sidebar.title("Navigation")
if st.sidebar.button("Visit AI Video Generator"):
    webbrowser.open_new_tab("https://pollo.ai/text-to-video")  # Pollo AI URL

compositor = ImageCompositor()

st.header("Upload Images")
image1 = st.file_uploader("Upload the first image", type=["jpg", "png"])
image2 = st.file_uploader("Upload the second image", type=["jpg", "png"])

if st.button("Composite Images"):
    if image1 and image2:
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp2:
            temp1.write(image1.read())
            temp2.write(image2.read())

            img1 = cv2.imread(temp1.name)
            img2 = cv2.imread(temp2.name)

            if img1 is not None and img2 is not None:
                result = compositor.composite_images(img1, img2)
                if result is not None:
                    st.image(result, caption="Composited Image", channels="BGR")

                    # Convert the result to a downloadable format
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(result_rgb)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0)

                    # Add download button
                    st.download_button(
                        label="Download Composited Image",
                        data=buffer,
                        file_name="composited_image.png",
                        mime="image/png"
                    )
            else:
                st.error("Error reading uploaded images.")
    else:
        st.error("Please upload both images.")

st.sidebar.info("Developed with ❤️ using Streamlit")
