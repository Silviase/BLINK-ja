import torch
from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from transformers import LlamaTokenizer
from PIL import Image

device_id = 0
device = f"cuda:{device_id}"
max_length = 512
model_id = "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1"
model = VideoBlipForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, ignore_mismatched_sizes=True
)
model = model.half()
model.eval()
model.to(device)

# prepare a processor
processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
processor.tokenizer = tokenizer


# inference 
def query_heron(image_paths, prompt):
    """
    Query the turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1 with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    images = [Image.open(image_path) for image_path in image_paths]
    text = f"##human: {prompt}\n##gpt: "

    # do preprocessing
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        truncation=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float16)

    # set eos token
    eos_token_id_list = [
        processor.tokenizer.pad_token_id,
        processor.tokenizer.eos_token_id,
        int(tokenizer.convert_tokens_to_ids("##"))
    ]

    # do inference
    with torch.no_grad():
        out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., eos_token_id=eos_token_id_list, no_repeat_ngram_size=2)
    generated_text = processor.tokenizer.batch_decode(out)
    
    return generated_text

    
if __name__ == "__main__":
    img_path = ["/nas64/silviase/Project/prj-blink-ja/submodule/BLINK_Benchmark/assets/teaser.png"]
    prompt = "この画像について説明してください。"
    print(query_heron(img_path, prompt))    