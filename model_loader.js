window.model = null;
export async function loadModel() {
    try {
        window.model = await tf.loadGraphModel('./tfjs_model/model.json');
        console.log('Model input shape:', window.model.inputs[0].shape);
        return window.model;
    } catch (err) {
        console.error('Model load failed:', err);
        alert('Failed to load model. Check console for details.');
    }
}