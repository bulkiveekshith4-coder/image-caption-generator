
const imageInput = document.getElementById("imageInput");

const chooseButton = document.getElementById("chooseButton");
const generateButton = document.getElementById("generateButton");

const preview = document.getElementById("preview");
const uploadArea = document.getElementById("uploadArea");

const removeButton = document.getElementById("removeButton");

const loading = document.getElementById("loading");
const resultBox = document.getElementById("resultBox");
const caption = document.getElementById("caption");


chooseButton.addEventListener("click", () => {
    imageInput.click();
});


imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image.");
        imageInput.value = "";
        return;
    }

    preview.src = URL.createObjectURL(file);

    preview.style.display = "block";
    uploadArea.style.display = "none";
    removeButton.style.display = "inline-block";

    generateButton.disabled = false;

    resultBox.style.display = "none";
});


removeButton.addEventListener("click", () => {

    imageInput.value = "";

    preview.removeAttribute("src");
    preview.style.display = "none";

    uploadArea.style.display = "block";
    removeButton.style.display = "none";

    generateButton.disabled = true;

    resultBox.style.display = "none";
});


generateButton.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) return;

    const formData = new FormData();

    formData.append("image", file);

    loading.style.display = "block";
    resultBox.style.display = "none";
    generateButton.disabled = true;

    try {

        const response = await fetch("/generate", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Caption generation failed");
        }

        caption.textContent = data.caption;

        resultBox.style.display = "block";

    } catch (error) {

        alert(error.message);

    } finally {

        loading.style.display = "none";
        generateButton.disabled = false;

    }

});
