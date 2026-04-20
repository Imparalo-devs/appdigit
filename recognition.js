export async function recognize(tensor) {
    if (!model) throw new Error("Model not loaded yet. Please wait for initialization.");
    return model.predict(tensor);
}