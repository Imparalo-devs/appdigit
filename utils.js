import * as tf from '@tensorflow/tfjs';

export function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor;
}

export function runInference(model, tensor) {
    return tf.tidy(() => model.predict(tensor));
}

export function displayResult(result) {
    const predictions = result.dataSync();
    const predictedDigit = predictions.indexOf(Math.max(...predictions));
    const confidence = Math.max(...predictions);
    document.getElementById('predictedDigit').textContent = predictedDigit.toString();
    document.getElementById('confidenceScore').textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
}