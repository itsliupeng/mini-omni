import os
import lightning as L
import torch
import time
from snac import SNAC
from litgpt import Tokenizer
from litgpt.utils import (
    num_parameters,
)
from litgpt.generate.base import (
    generate_AA,
    generate_ASR,
    generate_TA,
    generate_TT,
    generate_AT,
    generate_TA_BATCH,
    next_token_batch
)
import soundfile as sf
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import whisper
from tqdm import tqdm
from huggingface_hub import snapshot_download


torch.set_printoptions(sci_mode=False)


############################################################
device = "cuda:0"
out_dir = f"./output/{get_time_str()}"
ckpt_dir = f"/lp/models/mini-omni"
fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

task = ['A1A2', 'asr', "T1A2", "AA-BATCH", 'T1T2', 'AT']

# prepare test data
# TODO
test_audio_list = sorted(os.listdir('./data/samples'))
test_audio_list = [os.path.join('./data/samples', path) for path in test_audio_list]
test_audio_transcripts = [
    "What is your name?",
    "what are your hobbies?",
    "Do you like beijing",
    "How are you feeling today?",
    "what is the weather like today?",
]
test_text_list = [
    "What is your name?",
    "How are you feeling today?",
    "Can you describe your surroundings?",
    "What did you do yesterday?",
    "What is your favorite book and why?",
    "How do you make a cup of tea?",
    "What is the weather like today?",
    "Can you explain the concept of time?",
    "Can you tell me a joke?",
]



if __name__ == "__main__":
    mel, leng = load_audio(path)
    audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device)
    
    text = A1_A2(
        fabric,
        audio_feature,
        input_ids,
        leng,
        model,
        text_tokenizer,
        step,
        snacmodel,
        out_dir=out_dir,
    )
    print(f"input: {test_audio_transcripts[step]}")
    print(f"output: {text}")