import { recognize } from './recognition.js';
let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let predictedDigit = document.getElementById('predicted-digit');
let confidenceScore = document.getElementById('confidence-score');
let isDrawing = false;
let loading = true;

ctx.strokeStyle = '#e5a50a';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = '0';
}

function recognizeDraw(event) {
    console.log('.Button clicked at', new Date().toISOString());
    if (loading) return;
    loading = true;
    
    console.log('Current model state:', {
        loaded: !!window.model,
        modelType: window.model ? window.model.constructor.name : 'null'
    });
    
    const tensor = preprocessImage(canvas);
    console.log('Tensor created with shape:', tensor.shape);
    
    try {
        recognize(tensor).then((predictions) => {
            console.log('Predictions received:', predictions);
            const predictedDigitIndex = predictions.argMax(-1);
            const confidence = predictions.max(-1);
            
            predictions.data().then(data => {
                console.log('Raw prediction values:', Array.from(data));
            });
            console.log('Final prediction:', {
                digit: predictedDigitIndex,
                confidence: confidence
            });
            
            predictedDigit.textContent = predictedDigitIndex.toString();
            confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
            loading = false;
        }).catch((err) => {
            console.error('Error during recognition:', err, 'Stack:', err.stack);
            loading = false;
        });
    } catch (err) {
        console.error('Error calling recognize():', err, 'Tensor:', tensor);
        loading = false;
    }
}

function preprocessImage(canvas) {
    console.log('.Canvas raw data:', canvas.toDataURL());
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Debug canvas dopo resize
    const debugCanvas = document.createElement('canvas');
    debugCanvas.width = 100; debugCanvas.height = 100;
    const debugCtx = debugCanvas.getContext('2d');
    debugCtx.drawImage(tempCanvas, 0, 0, 100, 100);
    document.body.appendChild(debugCanvas); // Mostra immagine preprocessed
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    console.log('Pixel raw values:', Array.from(imageData.data).slice(0, 20)); // Primi 20 pixel
    
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const grayscale = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = grayscale;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor;
}

import { loadModel } from './model_loader.js';
loadModel().then(() => {
    console.log('✅ Model loaded successfully:', window.model);
    console.log('Model architecture:', JSON.stringify(window.model.weights, null, 2));
    document.getElementById('Recognize').disabled = false;
}).catch((err) => {
    console.error('❌ Model load failed:', err);
    alert('Could not load model. Check console for details.');
});
document.getElementById('Recognize').addEventListener('click', recognizeDraw);
document.getElementById('Clear').addEventListener('click', clearAll);
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);