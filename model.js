let model;
let loading = false;

async function loadModel() {
    if (loading) return;
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

async function runInference(tensor) {
    if (!model) await loadModel();
    return tf.tidy(() => model.predict(tensor));
}