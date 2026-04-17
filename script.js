function clearAll() {
    clearCanvas();
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceScore').textContent = '—';
}

function recognizeDraw() {
    runInference();
}