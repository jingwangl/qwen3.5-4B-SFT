from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/jingwangl/models/Qwen3.5-4B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# print(model)

messages = [
    {"role": "user", "content": "请输出一句简短的话，证明你已经在本地成功运行。"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

generated = outputs[0][inputs.input_ids.shape[1]:]
result = tokenizer.decode(generated, skip_special_tokens=True)

print("\n=== Model Output ===")
print(result)