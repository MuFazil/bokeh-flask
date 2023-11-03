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

// document
//   .getElementById("upload-button")
//   .addEventListener("change", function () {
//     const customUploadLabel = document.querySelector(".custom-file-upload");
//     const input = this;
//     customUploadLabel.textContent =
//       input.files.length > 0 ? input.files[0].name : "Choose an Image";
//   });
