// use.js - upload one document image and display the trained model's prediction.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const input = document.getElementById("document-file");
  const button = document.getElementById("predict-button");
  const message = document.getElementById("upload-message");
  const card = document.getElementById("prediction-card");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearChildren(message);
    clearChildren(document.getElementById("prediction-summary"));
    clearChildren(document.getElementById("prediction-scores"));
    clearChildren(document.getElementById("prediction-note"));
    card.hidden = true;
    const file = input.files[0];
    if (!file) {
      showBanner(message, "Choose a document image first.", "error");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      showBanner(message, "The image is larger than 15 MB. Choose a smaller file.", "error");
      return;
    }

    button.disabled = true;
    button.textContent = "Classifying…";
    try {
      const formData = new FormData();
      formData.append("document", file);
      const result = await apiCall("/api/predict", { method: "POST", body: formData });
      const summary = document.getElementById("prediction-summary");
      summary.appendChild(heading(result.predicted_class.replaceAll("_", " "), 2));
      summary.appendChild(paragraph(
        `Top score: ${(result.confidence * 100).toFixed(1)}% · ${result.filename}`,
        "verdict positive"));
      const scores = document.getElementById("prediction-scores");
      scores.appendChild(heading("Scores for all document types", 3));
      const list = document.createElement("div");
      list.className = "prediction-list";
      result.scores.forEach((item) => {
        const row = document.createElement("div");
        row.className = "prediction-row";
        const label = document.createElement("span");
        label.className = "prediction-label";
        label.textContent = item.class.replaceAll("_", " ");
        const track = document.createElement("div");
        track.className = "prediction-track";
        const fill = document.createElement("div");
        fill.className = "prediction-fill";
        fill.style.width = `${Math.max(0, Math.min(100, item.probability * 100))}%`;
        track.appendChild(fill);
        const value = document.createElement("span");
        value.className = "prediction-value";
        value.textContent = `${(item.probability * 100).toFixed(1)}%`;
        row.append(label, track, value);
        list.appendChild(row);
      });
      scores.appendChild(list);
      document.getElementById("prediction-note").appendChild(paragraph(
        `${result.model} · best checkpoint from epoch ${result.checkpoint_epoch}. ${result.note}`,
        "muted"));
      card.hidden = false;
    } catch (error) {
      showBanner(message, error.message, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Classify document";
    }
  });
});
