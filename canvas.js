export function initCanvas() {
    // No initialization needed
}

export function drawOnCanvas(ctx, previousX, previousY, x, y) {
    ctx.beginPath();
    ctx.moveTo(previousX, previousY);
    ctx.lineTo(x, y);
    ctx.stroke();
}