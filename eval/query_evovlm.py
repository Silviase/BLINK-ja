import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests

# 1. load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SakanaAI/EvoVLM-JP-v1-7B"
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_id)
model.to(device)


def query_evovlm(image_paths, prompt):
    """
    Query the evoVLMwith the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    # <image> represents the input image. Please make sure to put the token in your text.
    text = "<image>" * len(image_paths) + "\n" + prompt
    messages = [
        {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。"},
        {"role": "user", "content": text},
    ]
    inputs = processor.image_processor(images=images, return_tensors="pt")
    inputs["input_ids"] = processor.tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    )
    output_ids = model.generate(**inputs.to(device))
    output_ids = output_ids[:, inputs.input_ids.shape[1] :]
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text

    
if __name__ == "__main__":
    img_path = ["/nas64/silviase/Project/prj-blink-ja/submodule/BLINK_Benchmark/assets/teaser.png"]
    prompt = "この画像について説明してください。"
    print(query_evovlm(img_path, prompt))    