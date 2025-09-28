from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='flash_attention_2').to('cuda')
tokenizer = AutoTokenizer.from_pretrained(path)

x = "Hello, my name is"

inp = tokenizer(x, return_tensors='pt')

inp = {k:v.to(model.device) for k,v in inp.items()}

print('Generating')
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model.generate(**inp, min_new_tokens=256, do_sample=True, max_new_tokens=256)

print(tokenizer.batch_decode(out))