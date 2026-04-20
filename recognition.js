export async function recognize(tensor) {
    if (!window.model) {
        throw new Error('Model not loaded');
    }
    const predictions = window.model.execute(tensor);
    return predictions;
}