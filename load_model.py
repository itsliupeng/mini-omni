import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper

# audio_model = whisper.load_model("medium")

model_path = '/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_yi_6b_4m_bs1024_load_d1009_w0_banlance_loss_freeze_whisperm/test_yi_6b_4m_bs1024_load_d1009_w0_banlance_loss_freeze_whisperm/checkpoint/iter_0032000_hf'

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()


print("Done")