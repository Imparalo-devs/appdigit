let model;
let loading = false;

async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

document.addEventListener('DOMContentLoaded', () => {
    loadModel();
});