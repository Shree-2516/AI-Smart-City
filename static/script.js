// ===============================
// DOM ELEMENTS
// ===============================
const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const cameraBtn = document.getElementById("cameraBtn");

const cameraContainer = document.getElementById("cameraContainer");
const captureBtn = document.getElementById("captureBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");

const video = document.getElementById("camera");
const canvas = document.getElementById("captureCanvas");
const ctx = canvas.getContext("2d");

const loadingDiv = document.getElementById("loading");
const outputDiv = document.getElementById("output");
const outputImage = document.getElementById("outputImage");
const summaryText = document.getElementById("summaryText");
const severityBadge = document.getElementById("severityBadge");

let chartInstance = null;
let cameraStream = null;

// ===============================
// UPLOAD IMAGE
// ===============================
uploadBtn.addEventListener("click", () => {
    console.log("Upload button clicked");
    imageInput.click();
});

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    console.log("File selected:", file.name);

    const formData = new FormData();
    formData.append("image", file);

    sendToServer(formData);
});

// ===============================
// OPEN CAMERA (MANUAL MODE)
// ===============================
cameraBtn.addEventListener("click", () => {
    console.log("Camera button clicked");

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            cameraStream = stream;
            video.srcObject = stream;
            video.play();

            cameraContainer.classList.remove("d-none");
        })
        .catch(err => {
            console.error("Camera error:", err);
            alert("Camera access denied");
        });
});

// ===============================
// CAPTURE PHOTO (MANUAL)
// ===============================
captureBtn.addEventListener("click", () => {
    if (!cameraStream) return;

    console.log("Capture photo clicked");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Stop camera
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;

    cameraContainer.classList.add("d-none");

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("image", blob, "camera.jpg");
        sendToServer(formData);
    }, "image/jpeg");
});

// ===============================
// CLOSE CAMERA
// ===============================
closeCameraBtn.addEventListener("click", () => {
    console.log("Close camera clicked");

    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }

    cameraContainer.classList.add("d-none");
});

// ===============================
// SEND IMAGE TO FLASK
// ===============================
function sendToServer(formData) {
    console.log("Sending image to server...");

    loadingDiv.classList.remove("d-none");
    outputDiv.classList.add("d-none");

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        console.log("Server response:", data);

        loadingDiv.classList.add("d-none");
        outputDiv.classList.remove("d-none");

        // Output image
        outputImage.src = "data:image/png;base64," + data.image;

        // Summary
        summaryText.innerText = formatSummary(data.summary);

        // Severity (from backend)
        updateSeverityFromServer(data.severity);

        // Chart
        drawChart(data.summary);
    })
    .catch(err => {
        console.error("Prediction error:", err);
        loadingDiv.classList.add("d-none");
        alert("Error during prediction");
    });
}

// ===============================
// SUMMARY TEXT
// ===============================
function formatSummary(summary) {
    if (!summary || Object.keys(summary).length === 0) {
        return "No major issues detected.";
    }

    return "Detected Issues: " +
        Object.entries(summary)
            .map(([k, v]) => `${v} ${k}`)
            .join(", ");
}

// ===============================
// SEVERITY BADGE
// ===============================
function updateSeverityFromServer(severity) {
    severityBadge.className = "severity";

    severityBadge.innerText = "Severity: " + severity;

    if (severity === "Low") {
        severityBadge.classList.add("low");
    } else if (severity === "Medium") {
        severityBadge.classList.add("medium");
    } else {
        severityBadge.classList.add("high");
    }

    severityBadge.classList.remove("d-none");
}

// ===============================
// PIE CHART
// ===============================
function drawChart(summary) {
    const ctxChart = document.getElementById("chartCanvas").getContext("2d");

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctxChart, {
        type: "pie",
        data: {
            labels: Object.keys(summary),
            datasets: [{
                data: Object.values(summary),
                backgroundColor: [
                    "#0d6efd",
                    "#dc3545",
                    "#198754",
                    "#ffc107"
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}
