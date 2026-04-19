document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("spam-form");
    const messageInput = document.getElementById("message-input");
    const submitBtn = document.getElementById("analyze-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const loader = submitBtn.querySelector(".loader");
    
    const resultContainer = document.getElementById("result-container");
    const resultCard = document.querySelector(".result-card");
    const resultTitle = document.getElementById("result-title");
    const resultDescription = document.getElementById("result-description");
    const confidenceBadge = document.getElementById("confidence-badge");
    const confidenceBar = document.getElementById("confidence-bar");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // UI Loading State
        setLoading(true);
        hideResult();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error("Server error or model unavailable.");
            }

            const data = await response.json();
            showResult(data);
        } catch (error) {
            console.error("Error during prediction:", error);
            showError("Failed to analyze the message. Please try again later.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.classList.add("hidden");
            loader.classList.remove("hidden");
        } else {
            submitBtn.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");
        }
    }

    function hideResult() {
        resultContainer.classList.add("hidden");
        // Reset classes
        resultCard.className = "result-card";
        confidenceBar.style.width = "0%";
    }

    function showResult(data) {
        const { prediction, confidence } = data;
        const confidencePercent = (confidence * 100).toFixed(1);
        
        resultContainer.classList.remove("hidden");
        
        // Let UI flow settle slightly before animating bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;
        }, 50);
        
        confidenceBadge.textContent = `${confidencePercent}% Confidence`;

        if (prediction === "spam") {
            resultCard.classList.add("is-spam");
            resultTitle.textContent = "Spam Detected";
            resultDescription.textContent = "Warning! This message exhibits characteristics common to spam, phishing, or unsolicited mass mailings.";
        } else {
            resultCard.classList.add("is-ham");
            resultTitle.textContent = "Looks Safe";
            resultDescription.textContent = "This message appears to be legitimate and does not match our known spam signatures.";
        }
    }

    function showError(message) {
        resultContainer.classList.remove("hidden");
        resultCard.className = "result-card is-spam"; // Red border
        resultTitle.textContent = "Error";
        resultDescription.textContent = message;
        confidenceBadge.textContent = "N/A";
        confidenceBar.style.width = "0%";
    }
});
