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

            image = cv2.imread("static/uploaded_image.jpg", cv2.IMREAD_GRAYSCALE)
            # Set a threshold value
            threshold_value = 128  # You can adjust this threshold as needed
            # Apply binary thresholding
            ret, binary_image = cv2.threshold(
                image, threshold_value, 255, cv2.THRESH_BINARY
            )
            # Save or display the binarized image
            cv2.imwrite("static/processed_image.jpg", binary_image)

            session["uploaded_image"] = "static/uploaded_image.jpg"

            # Redirect to the index page after processing

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
