from fastai.vision.all import *
import gradio as gr

learn = load_learner('bear_classifier.pkl')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {str(learn.dls.vocab[i]): float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Bear Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))