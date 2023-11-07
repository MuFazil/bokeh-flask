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
import torch

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
            import numpy as np

            # download MiDaS model
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.to("cpu")
            midas.eval()

            # input tranformational pipeline
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = transforms.small_transform

            resize_percentage = 30
            width = int(image.shape[1] * resize_percentage / 100)
            height = int(image.shape[0] * resize_percentage / 100)
            print(width, height)
            image = cv2.resize(image, (width, height))

            cv2.imshow("Given image", image)

            # Tranform image to input for midas
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imagebatch = transform(image).to("cpu")

            # make a prediction
            with torch.no_grad():
                prediction = midas(imagebatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                output = prediction.to("cpu").numpy()
                output_norm = cv2.normalize(
                    output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

            # thresholding the image
            ret, mask = cv2.threshold(
                output_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            mask_fore = cv2.bitwise_not(mask)

            # creating masks and finding foreground and background separetely
            bokeh_image = np.copy(image)
            foreground_sub = np.copy(image)
            foreground_sub = cv2.bitwise_and(
                foreground_sub, foreground_sub, mask=mask_fore
            )
            bokeh_image = cv2.bitwise_and(bokeh_image, bokeh_image, mask=mask)

            # blurring the background of the image
            bokeh_image = cv2.GaussianBlur(bokeh_image, (5, 5), 5)

            # adding the blurred background to the foreground resulting in the bokeh effect
            output_image = cv2.add(bokeh_image, foreground_sub)

            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            # Save or display the binarized image
            cv2.imwrite("static/processed_image.jpg", output_image)
            # cv2.imshow("Bokeh image", bokeh_image)
            session["uploaded_image"] = "static/uploaded_image.jpg"

            # Redirect to the index page after processing

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
