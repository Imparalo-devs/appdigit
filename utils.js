function preprocessImage(canvas) {
    let ctx = canvas.getContext('2d');
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let pixels = imageData.data;
    let grayscalePixels = new Uint8Array(pixels.length / 4);
    for (let i = 0; i < pixels.length; i += 4) {
        let gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    let tensor = tf.tensor3d(grayscalePixels, [28, 28, 1], 'uint8');
    return tensor.toFloat().div(255);
}

function displayResult(result) {
    let predictedDigit = result.argMax(1).dataSync()[0];
    let confidence = result.max(1).dataSync()[0];
    document.getElementById('predictedDigit').textContent = predictedDigit.toString();
    document.getElementById('confidenceScore').textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
}