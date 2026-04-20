export async function recognize(tensor) {
    if (!window.model) throw new Error('Model not loaded');
    const predictions = await window.model.execute(tensor);
    const data = await predictions.data();
    return data;
}