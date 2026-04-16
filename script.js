import { loadModel } from './model.js';
import { initCanvas, drawOnCanvas } from './canvas.js';
import { preprocessImage, runInference, displayResult } from './utils.js';

let model;
let canvas;
let ctx;
let drawing = false;
let lastX, lastY;

loadModel().then((loadedModel) => {
    model = loadedModel;
    initCanvas();
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    canvas.addEventListener('pointerdown', (e) => {
        drawing = true;
        lastX = e.clientX - canvas.getBoundingClientRect().left;
        lastY = e.clientY - canvas.getBoundingClientRect().top;
    });
    canvas.addEventListener('pointermove', (e) => {
        if (drawing && e.buttons === 1) {
            const x = e.clientX - canvas.getBoundingClientRect().left;
            const y = e.clientY - canvas.getBoundingClientRect().top;
            drawOnCanvas(ctx, lastX, lastY, x, y);
            lastX = x;
            lastY = y;
        }
    });
    canvas.addEventListener('pointerup', () => {
        drawing = false;
        recognizeDraw();
    });
});

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').textContent = 'N/A';
    document.getElementById('confidenceScore').textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    const tensor = preprocessImage(canvas);
    runInference(model, tensor).then((result) => {
        displayResult(result);
    });
}