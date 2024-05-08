import torch
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from PIL import Image
import requests

# load model
model = AutoModelForVision2Seq.from_pretrained("stabilityai/japanese-instructblip-alpha", trust_remote_code=True)
processor = BlipImageProcessor.from_pretrained("stabilityai/japanese-instructblip-alpha")
tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p


def concat_images_horizontally_with_margin(image_paths, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - new_image: PIL Image, the concatenated image.
    """
    images = [Image.open(filename) for filename in image_paths]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one

    return new_image

def query_japanese_instructblip_alpha(image_paths, prompt):
    """
    Query the evoVLMwith the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    # prepare inputs
    
    # multiple images
    # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    image = concat_images_horizontally_with_margin(image_paths)
    inputs = processor(images=image, return_tensors="pt")
    
    prompt = build_prompt(prompt)
    text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
    text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
    
    inputs.update(text_encoding)

    # generate
    outputs = model.generate(
        **inputs.to(device, dtype=model.dtype),
        num_beams=5,
        max_new_tokens=64,
        min_length=1,
    )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    return generated_text



