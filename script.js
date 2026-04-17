import { loadModel } from './model.js';
import { initCanvas, handlePointerMove } from './canvas.js';
import { preprocessImage, displayResult } from './utils.js';

let model;
let canvas;
let ctx;
let drawing = false;

loadModel().then((loadedModel) => {
    model = loadedModel;
    const { canvas: canvasElement, ctx: context } = initCanvas();
    canvas = canvasElement;
    ctx = context;
    document.addEventListener('pointerup', () => {
        recognizeDraw();
    });
});

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').textContent = 'N/A';
    document.getElementById('confidenceScore').textContent = 'Confidence score -> 0';
}

async function recognizeDraw() {
    const tensor = preprocessImage(canvas);
    const predictions = model.predict(tensor);
    const predictedDigit = await predictions.argMax().data();
    const maxValue = await predictions.max().data();
    const confidence = maxValue[0] * 100 + '%';
    displayResult(predictedDigit[0], confidence);
}