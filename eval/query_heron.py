import torch
from heron.models.git_llm.git_japanese_stablelm_alpha import GitJapaneseStableLMAlphaForCausalLM
from transformers import AutoProcessor, LlamaTokenizer
from PIL import Image

device = f"cuda"
model_id = "turing-motors/heron-chat-git-ja-stablelm-base-7b-v1"
model = GitJapaneseStableLMAlphaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, ignore_mismatched_sizes=True
)
model.eval()
model.to(device)

# prepare a processor
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = LlamaTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1",
    padding_side="right",
    additional_special_tokens=["▁▁"],
)
processor.tokenizer = tokenizer

# inference 
def query_heron(image_paths, prompt):
    """
    Query the turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1 with the prompt and a list of image paths.

    Parameters:
    - image_paths: List of Strings, the path to the images.
    - prompt: String, the prompt.
    """
    
    # prepare inputs
    # images = [Image.open(image_path) for image_path in image_paths]
    images = concat_images_horizontally_with_margin(image_paths)

    text = f"##human: {prompt}?\n##gpt: "

    # do preprocessing
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        truncation=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # do inference
    with torch.no_grad():
        out = model.generate(**inputs, max_length=256, do_sample=False, temperature=0., no_repeat_ngram_size=2)

    # print result
    generated_text = processor.tokenizer.batch_decode(out)[0].split("##gpt: ")[1].replace("<|endoftext|>", "")
    return generated_text

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
    
if __name__ == "__main__":
    img_path = ["/nas64/silviase/Project/prj-blink-ja/BLINK_Benchmark/assets/test.jpeg"]
    prompts = [
        "青い浮き輪は何個ありますか？",
        "How many blue floats are there? Select from the following choices. (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 一つ選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選んで答えなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選んで答えなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選びなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から答えなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から答えなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から選びなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
    ]
    
    for prompt in prompts:
        print(query_heron(img_path, prompt))    