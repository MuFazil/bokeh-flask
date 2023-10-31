import cv2

# Load the image
image = cv2.imread("D:/bokeh-flask/demo.jpg")

# Load the pre-trained model for background segmentation (grabCut)
bg_model = cv2.bgsegm.createBackgroundSubtractorMOG()

# Apply background segmentation to the image
mask = bg_model.apply(image)

# Invert the mask to get the foreground
mask = cv2.bitwise_not(mask)

# Apply Gaussian blur to the background
blurred_background = cv2.GaussianBlur(image, (15, 15), 0)

# Create the bokeh effect by combining the foreground and blurred background
bokeh_image = cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(
    blurred_background, blurred_background, mask=~mask
)

# Save or display the bokeh effect image
cv2.imwrite("bokeh_image.jpg", bokeh_image)

# If you want to display the image using OpenCV's imshow
# cv2.imshow("Bokeh Image", bokeh_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
