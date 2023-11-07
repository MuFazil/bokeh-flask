document
  .getElementById("upload-button")
  .addEventListener("change", function (e) {
    const uploadedImage = document.getElementById("uploaded-image");
    const file = e.target.files[0];

    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        uploadedImage.src = e.target.result;
      };
      reader.readAsDataURL(file);
    } else {
      uploadedImage.src =
        "{{ url_for('static', filename='uploaded_image.jpg') }}";
    }
  });
