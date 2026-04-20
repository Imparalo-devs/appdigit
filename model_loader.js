export async function loadModel() {
    try {
        const model = await tf.loadGraphModel('/tfjs_model/model.json'); // Use /tfjs_model/ prefix
        return model;
    } catch (err) {
        console.error('Model load failed:', err);
        throw err;
    }
}