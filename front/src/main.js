const uploadPage = document.getElementById("upload-page");
const resultPage = document.getElementById("result-page");
const resultImage = document.getElementById("result-image");
const backButton = document.getElementById("backButton");
const statusText = document.getElementById("uploadForm_Hint");
const uploadInput = document.querySelector(".form-upload__input");
const dropFileZone = document.querySelector(".upload-zone_dragover");

const uploadUrl = "http://0.0.0.0:8000/detect";

// Initialize event listeners
function init() {
  // Drag and drop handlers
  ["dragover", "drop"].forEach(event => {
    document.addEventListener(event, evt => {
      evt.preventDefault();
    });
  });

  dropFileZone.addEventListener("dragenter", () => {
    dropFileZone.classList.add("_active");
  });

  dropFileZone.addEventListener("dragleave", () => {
    dropFileZone.classList.remove("_active");
  });

  dropFileZone.addEventListener("drop", e => {
    e.preventDefault();
    dropFileZone.classList.remove("_active");
    handleFile(e.dataTransfer?.files[0]);
  });

  uploadInput.addEventListener("change", () => {
    handleFile(uploadInput.files?.[0]);
  });

  backButton.addEventListener("click", () => {
    showUploadPage();
  });
}

function handleFile(file) {
  if (!file?.type?.startsWith("image/")) {
    setStatus("Можно загружать только изображения");
    return;
  }
  processingUploadFile(file);
}

function setStatus(text) {
  statusText.textContent = text;
}

function showResultPage() {
  uploadPage.style.display = "none";
  resultPage.style.display = "flex";
}

function showUploadPage() {
  resultPage.style.display = "none";
  uploadPage.style.display = "flex";
  setStatus("");
  uploadInput.value = ""; // Clear file input
}

async function processingUploadFile(file) {
  setStatus("Загрузка...");
  
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(uploadUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Ошибка сервера");
    }

    const blob = await response.blob();
    resultImage.src = URL.createObjectURL(blob);
    showResultPage();
    setStatus("");
  } catch (error) {
    setStatus(error.message || "Ошибка при загрузке файла");
    console.error("Upload error:", error);
  }
}

async function downloadReport(format) {
  try {
    const response = await fetch(`/report/${format}`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection_report.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading report:', error);
    alert('Failed to download report');
  }
}

// Initialize the app
init();