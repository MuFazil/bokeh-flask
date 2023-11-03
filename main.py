from flask import (
    Flask,
    render_template,
    Response,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)
import cv2

app = Flask(__name__)
app.static_folder = "static"
app.secret_key = "bellingham"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_image = request.files["image"]
        if uploaded_image:
            # Save the uploaded image to a temporary location
            uploaded_image.save("static/uploaded_image.jpg")
            # Perform image processing with OpenCV here
            # The processed image should replace the placeholder-image.png

            image = cv2.imread("static/uploaded_image.jpg")
            # Set a threshold value
            import numpy as np
            import matplotlib.pyplot as plt

            resize_percentage = 40
            width = int(image.shape[1] * resize_percentage / 100)
            height = int(image.shape[0] * resize_percentage / 100)
            image = cv2.resize(image, (width, height))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_values = image.reshape((-1, 3))

            pixel_values = np.float32(pixel_values)

            # define stopping criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

            # number of clusters (K)
            k = 2
            _, labels, (centers) = cv2.kmeans(
                pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            # convert back to 8 bit values
            centers = np.uint8(centers)

            # flatten the labels array
            labels = labels.flatten()

            # convert all pixels to the color of the centroids
            segmented_image = centers[labels.flatten()]

            # reshape back to the original image dimension
            segmented_image = segmented_image.reshape(image.shape)

            cluster = range(10)
            for i in cluster:
                masked_image = np.copy(image)
                masked_image = masked_image.reshape((-1, 3))
                masked_image[labels == i] = [0, 0, 0]
                # convert back to original shape
                masked_image = masked_image.reshape(image.shape)

            # disable the non-required clusters(background segments)
            masked_image = np.copy(image)
            # convert to the shape of a vector of pixel values
            masked_image = masked_image.reshape((-1, 3))
            # color (i.e cluster) to disable
            cluster = [1]
            for i in cluster:
                masked_image[labels == i] = [0, 0, 0]
            # convert back to original shape
            masked_image = masked_image.reshape(image.shape)

            # convert the image back to BGR format
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

            # creating a mask from the foreground image for the background
            gray_mask = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY_INV)

            # historical_background = cv2.bitwise_and(image, image, mask=mask)
            historical_background = cv2.GaussianBlur(
                image, (25, 25), 0
            )  # Default kernel size

            # Combine your resized image (foreground) and the historical background
            output_image = cv2.add(masked_image, historical_background)

            # Save or display the binarized image
            cv2.imwrite("static/processed_image.jpg", output_image)

            session["uploaded_image"] = "static/uploaded_image.jpg"

            # Redirect to the index page after processing

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
