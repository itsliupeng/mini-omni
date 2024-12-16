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
from utils.snac_utils import reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import whisper
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_printoptions(sci_mode=False)



# TODO
text_vocabsize = 64000
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4
_tts_t = text_vocabsize + 5

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4
_tts = audio_vocabsize + 5



def layershift(input_id, layer, stride=padded_audio_vocabsize, shift=padded_text_vocabsize):
    return input_id + shift + layer * stride


def get_input_ids_TA(text, text_tokenizer):
    input_ids_item = [[] for _ in range(8)]
    text_tokens = text_tokenizer.encode(text)
    for i in range(7):
        input_ids_item[i] = [layershift(_pad_a, i)] * (len(text_tokens) + 2) + [
            layershift(_answer_a, i)
        ]
        input_ids_item[i] = torch.tensor(input_ids_item[i]).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens.tolist() + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    return input_ids_item

def get_input_ids_TTS(text, text_tokenizer):
    input_ids_item = [[] for _ in range(8)]
    text_tokens = text_tokenizer.encode(text)
    for i in range(7):
        input_ids_item[i] = [layershift(_input_a, i)] + [layershift(_pad_a, i)] * len(text_tokens) + [layershift(_eoa, i), layershift(_tts, i)]
        input_ids_item[i] = torch.tensor(input_ids_item[i]).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens.tolist() + [_eot] + [_tts_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    return input_ids_item

def get_input_ids_TT(text, text_tokenizer):
    input_ids_item = [[] for i in range(8)]
    text_tokens = text_tokenizer.encode(text).tolist()
    
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)
    for i in range(7):
        input_ids_item[i] = input_ids_item[-1]

    return input_ids_item


def get_input_ids_whisper(
    mel, leng, whispermodel, device, 
    special_token_a=_answer_a, special_token_t=_answer_t, text=None, text_tokenizer=None
):

    if text and text_tokenizer:
        text_tokens = text_tokenizer.encode(text).tolist()
    else:
        text_tokens = None
    
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        # audio_feature = whispermodel.encoder(mel)[0][:leng]
        audio_feature = whispermodel.encoder(mel)[0][:leng]

    T = audio_feature.size(0)
    input_ids = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    
    if text_tokens:
        assert len(text_tokens) <= T
        input_id_T = torch.tensor([_input_t] +  text_tokens + [_pad_t] * (T-len(text_tokens)) + [_eot, special_token_t])
    else:
        input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return audio_feature.unsqueeze(0), input_ids


def get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device):
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
    T = audio_feature.size(0)
    input_ids_AA = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_answer_a, i)]
        input_ids_AA.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AA.append(input_id_T)

    input_ids_AT = [] 
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_pad_a, i)]
        input_ids_AT.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AT.append(input_id_T)

    input_ids = [input_ids_AA, input_ids_AT]
    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):
        for j in range(8):
            stacked_inputids[j].append(input_ids[i][j])
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]
    return torch.stack([audio_feature, audio_feature]), stacked_inputids


def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1


def A1_A2_batch(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                snacmodel, out_dir=None):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=2)
    tokenlist = generate_TA_BATCH(
        model,
        audio_feature,
        input_ids,
        [leng, leng],
        ["A1A2", "A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    text_tokenlist = tokenlist[-1]
    if text_vocabsize in text_tokenlist:
        text_tokenlist = text_tokenlist[: text_tokenlist.index(text_vocabsize)]
    text = text_tokenizer.decode(torch.tensor(text_tokenlist)).strip()

    audio_tokenlist = tokenlist[:-1]
    audiolist = reconscruct_snac(audio_tokenlist)
    audio = reconstruct_tensors(audiolist)
    if out_dir is None:
        out_dir = "./output/default/A1-A2-batch"
    else:
        out_dir = out_dir + "/A1-A2-batch"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    # model.clear_kv_cache()
    return text


def A1_T2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_AT(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["AT"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_A2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_AA(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
        layershift_shift=padded_text_vocabsize,
    )
    
    audiolist = reconscruct_snac(tokenlist)
    tokenlist = tokenlist[-1]
    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    if out_dir is None:
        out_dir = "./output/default/A1-A2"
    else:
        out_dir = out_dir + "/A1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(audiolist) == 0:
        return ""

    audio = reconstruct_tensors(audiolist)
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_T1(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_ASR(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T1"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_A2(fabric, input_ids, model, text_tokenizer, step,
          snacmodel, out_dir=None):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_TA(
        model,
        None,
        input_ids,
        None,
        ["T1A2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
        layershift_shift=padded_text_vocabsize,
    )

    audiolist = reconscruct_snac(tokenlist)
    tokenlist = tokenlist[-1]

    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    audio = reconstruct_tensors(audiolist)
    if out_dir is None:
        out_dir = "./output/default/T1-A2"
    else:
        out_dir = out_dir + "/T1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_T2(fabric, input_ids, model, text_tokenizer, step):

    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_TT(
        model,
        None,
        input_ids,
        None,
        ["T1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

    
def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("medium").to(device)
    text_tokenizer = Tokenizer("/lp/models/Yi-6B")
    # fabric = L.Fabric(devices=1, strategy="auto")
    # config = Config.from_file(ckpt_dir + "/model_config.yaml")
    # config.post_adapter = False

    # with fabric.init_module(empty_init=False):
    if True:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            device_map="cpu",
            torch_dtype=torch.float,
            trust_remote_code=True
        )

    # model = fabric.setup(model)
    # state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    # model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # return None, model, text_tokenizer, snacmodel, model.audio_model
    return None, model, text_tokenizer, snacmodel, whispermodel


    
def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

    
class OmniInference:

    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)

    def warm_up(self, sample='./data/samples/output1.wav'):
        for _ in self.run_AT_batch_stream(sample):
            pass

    @torch.inference_mode()
    def run_AT_batch_stream(self, 
                            audio_path, 
                            stream_stride=4,
                            max_returned_tokens=2048, 
                            temperature=0.9, 
                            top_k=1, 
                            top_p=1.0,
                            eos_id_a=_eoa,
                            eos_id_t=_eot,
        ):

        assert os.path.exists(audio_path), f"audio file {audio_path} not found"
        model = self.model

        with self.fabric.init_tensor():
            model.set_kv_cache(batch_size=2)

        mel, leng = load_audio(audio_path)
        audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, self.whispermodel, self.device)
        T = input_ids[0].size(1)
        device = input_ids[0].device

        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than audio length {T}"

        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        input_pos = torch.tensor([T], device=device)
        list_output = [[] for i in range(8)]
        tokens_A, token_T = next_token_batch(
            model,
            audio_feature.to(torch.float32).to(model.device),
            input_ids,
            [T - 3, T - 3],
            ["A1T2", "A1T2"],
            input_pos=torch.arange(0, T, device=device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * padded_audio_vocabsize
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=device))
            model_input_ids[i] = torch.stack(model_input_ids[i])

        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        text_end = False
        index = 1
        nums_generate = stream_stride
        begin_generate = False
        current_index = 0
        for _ in tqdm(range(2, max_returned_tokens - T + 1)):
            tokens_A, token_T = next_token_batch(
                model,
                None,
                model_input_ids,
                None,
                None,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if text_end:
                token_T = torch.tensor([_pad_t], device=device)

            if tokens_A[-1] == eos_id_a:
                break

            if token_T == eos_id_t:
                text_end = True

            for i in range(7):
                list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])

            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() +padded_text_vocabsize + i * padded_audio_vocabsize
                model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
                model_input_ids[i].append(
                    torch.tensor([layershift(4097, i)], device=device)
                )
                model_input_ids[i] = torch.stack(model_input_ids[i])

            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            if index == 7:
                begin_generate = True

            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    # import ipdb; ipdb.set_trace()
                    snac = get_snac(list_output, index, nums_generate)
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)
                    yield audio_stream

            input_pos = input_pos.add_(1)
            index += 1
        text = self.text_tokenizer.decode(torch.tensor(list_output[-1]))
        print(f"text output: {text}")
        model.clear_kv_cache()
        return list_output


def test_infer():
    device = "cuda:0"
    out_dir = f"./output/{get_time_str()}"
    # ckpt_dir = f"/lp/models/mini-omni"
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_4m_bs1024_load_wm_freeze_llm_extra_d1021/checkpoint/iter_0002400_hf"
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_4m_bs1024_load_wm_freeze_llm_extra_d1021/checkpoint/iter_0003200_hf"
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_4m_bs1024_load_wm_freeze_llm_extra_d1021/checkpoint/iter_0003200_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_8m_bs1024_load_wm_asr_fix/checkpoint/iter_0010554_hf_B"
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_4m_bs1024_load_wm_asr_pool_d1026_shuffle_size/checkpoint/iter_0010554_hf_pool"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio/yi_6b_8m_bs1024_load_wm_asr_pool_d1026_shuffle_size_A_sr/checkpoint/iter_0010554_hf_pool"
    
    # AA AT
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_4m_bs512_amode_t_proj_llm_extra_cg4_d1030/checkpoint/iter_0008000_hf"
    # ckpt_dir = "/gpfs/public/pretrain/liupeng/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_4m_bs512_amode_t_proj_llm_extra_cg4_d1030/checkpoint/iter_0024000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_4m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030/checkpoint/iter_0032000_hf" #
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_4m_bs512_onlyATT_t_proj_llm_extra_cg4_d1030/checkpoint/iter_0032000_hf" #
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030_from_scratch/checkpoint/iter_0060000_hf" # 20241104_071400
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_4m_bs512_aatmode_t_proj_llm_extra_cg4_d1030_A/checkpoint/iter_0038000_hf" # 20241104_080525
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030/checkpoint/iter_0052000_hf" # 20241104_084504
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030_C_ATA/checkpoint/iter_0018000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030_C_onlyATA/checkpoint/iter_0040000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030_C_onlyATA/checkpoint/iter_0086000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_8m_bs512_4aatmode_t_proj_llm_extra_cg4_d1030_C_ATA/checkpoint/iter_0080000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_16m_bs512_tts_ta8_quora/checkpoint/iter_0066000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_2m_bs2k_tts_ta0/checkpoint/iter_0010000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_2m_bs2k_tts_ta0/checkpoint/iter_0004000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_4m_bs2k_tts_ta8_quora_fllm/checkpoint/iter_0043000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_4m_bs2k_tts_ta8_quora_tloss/checkpoint/iter_0002000_hf"

    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_4m_bs2k_tts8_f_d1204_freezellm_trainextrawe_librilight/checkpoint/iter_0000600_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_4m_bs2k_tts8_f_d1204_freezellm_trainextrawe/checkpoint/iter_0001000_hf"
    ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/test_audio_tts/yi6b_bs1k_tts8_f_d1204_trainextrawe_librilight_quora_zhihu/checkpoint/iter_0009000_hf"
    
    # if not os.path.exists(ckpt_dir):
    #     print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
    #     download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

    # task = ['A1A2', 'asr', "T1A2", "AA-BATCH", 'T1T2', 'AT']
    # task = ["AA-BATCH"]
    # task = ["AT"]
    # task = ["A1A2"]
    # task = ['T1A2']
    task = ["tts"]
    print(f"task: {task}")
    # task = ["A1A2"]

    # prepare test data
    # TODO
    test_audio_list = sorted(os.listdir('./data/samples'))
    test_audio_list = [os.path.join('./data/samples', path) for path in test_audio_list]
    test_audio_transcripts = [
        "but the owl is not a burglar he is the friend of man there is no other bird that does the farmer so much good as the owl the owl comes out in the dark to get the small animals that are out at that time stealing things from the farmer",
        "What is your name?",
        "what are your hobbies?",
        "Do you like beijing",
        "How are you feeling today?",
        "what is the weather like today?",
        "a gentleman living in the west when there was so much damage done by grasshoppers found that the owls were living on them and not eating much of any other kind of food the only way he could tell what the owls had for supper was to shoot an owl once in awhile and see what was in its stomach"
    ]
    test_text_list = [
        # "Nope",
        "Once upon a time in a distant land,",
        "How are you feeling today?",
        "Can you describe your surroundings?",
        "What did you do yesterday?",
        "What is your favorite book and why?",
        "How do you make a cup of tea?",
        "What is the weather like today?",
        "Can you explain the concept of time?",
        "Can you tell me a joke?",
        "Nope"
    ]

    tts_text_list = [
        "Take along one sheet of paper from the hospital or doctor to prove you are not scamming. Good luck to you; you have a tough row to hoe, but it will be worth it.",
        "Your mum is doing for you what she thinks is best.Unfortunately, like all parents, it's hard when to know you stop and your child begins.As you get older you will change regardless and she will feel she has lost control.Comfort her by saying you will never leave, your heart is always with her wherever you go.That God has chosen your life and it's not the same as what she thinks.That what works for her, doesn't work for everyone else, as everyone is different.Reassure her she is doing a great job."
        "Map out your hours!Everyone is given the same twenty four hours in a day but the key to not getting burnt out is to plan ahead.Make rough to do lists for the day and goals for the week",
        "As a student, i know it's very difficult to balance personal or social life with studies as ca is a very long journey with vast syllabus. The quantum of syllabus makes us dull to enjoy other things in life.Like everytime you get a chance to enjoy, you always think about completion of syllabus and time is very limited but that make us monotonous. Monotonous work make us inefficient and inefficiency limits our scope of achieving targets and when the targets not get finished, we start doubting self.",
        "面试、重大场合的时候,让自己闪亮一下,又何乐而不为呢。这比长高可简单多了。不化妆的女生其实也很美。自然清纯,悠然自得有些女生化妆反而会显得老态,难以保持清纯的样子,不化妆显得更自然,更美。不化妆不代表不对皮肤进行护理,化妆对皮肤的负担很重",
        "之前由于院系大调整从浙大拆了出去,这些高校从弱小到壮大,浙江大学后来又将这些合并成一个巨无霸高校,就有人调侃是其是坐享其成。吐槽点:各种各样武汉大学和某些明星一样,天生招黑体质,不管做什么都有人黑。至于原因,有的说它是发展太快"
    ]


    # LOAD MODEL
    with torch.no_grad():
        if "A1A2" in task:
            print("===============================================================")
            print("                       testing A1A2")
            print("===============================================================")
            step = 0
            for idx, path in enumerate(test_audio_list):
                # if idx < 1:
                #     continue
                try:
                    mel, leng = load_audio(path)
                    audio_feature, input_ids = get_input_ids_whisper(
                        mel, leng, whispermodel, device, 
                        special_token_a=_answer_a, special_token_t=_answer_t
                    )
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
                    step += 1
                    print(
                        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                    )
                except Exception as e:
                    raise e
                    print(f"[error] {e} failed to process {path}")
            print("===============================================================")

        if 'asr' in task:
            print("===============================================================")
            print("                       testing asr")
            print("===============================================================")

            index = 0
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                # audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_pad_a, special_token_t=_answer_t)
                audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_pad_a, special_token_t=_asr)
                output = A1_T1(fabric, audio_feature, input_ids ,leng, model, text_tokenizer, index).lower().replace(',','').replace('.','').replace('?','')
                print(f"audio_path: {path}")
                print(f"audio transcript: {test_audio_transcripts[index]}")
                print(f"asr output: {output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                index += 1

        if "tts" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1A2")
            print("===============================================================")
            for idx, text in enumerate(tts_text_list):
                input_ids = get_input_ids_TTS(text, text_tokenizer)
                text_output = T1_A2(fabric, input_ids, model, text_tokenizer, step,
                                    snacmodel, out_dir=out_dir)
                print(f"-------- idx: {idx} ---------")
                print(f"input: {text}")
                print(f"output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                step += 1
            print("===============================================================")

        if "T1A2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1A2")
            print("===============================================================")
            for text in test_text_list:
                input_ids = get_input_ids_TA(text, text_tokenizer)
                text_output = T1_A2(fabric, input_ids, model, text_tokenizer, step,
                                    snacmodel, out_dir=out_dir)
                print(f"input: {text}")
                print(f"output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                step += 1
            print("===============================================================")

        if "T1T2" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing T1T2")
            print("===============================================================")

            for text in test_text_list:
                input_ids = get_input_ids_TT(text, text_tokenizer)
                text_output = T1_T2(fabric, input_ids, model, text_tokenizer, step)
                print(f" Input: {text}")
                print(f"Output: {text_output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AT" in task:
            print("===============================================================")
            print("                       testing A1T2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(
                    mel, leng, whispermodel, device, 
                    special_token_a=_pad_a, special_token_t=_answer_t
                )
                text = A1_T2(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step
                )
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        if "AA-BATCH" in task:
            print("===============================================================")
            print("                       testing A1A2-BATCH")
            print("===============================================================")
            step = 0
            for idx, path in enumerate(test_audio_list):
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device)
                text = A1_A2_batch(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step,
                    snacmodel, out_dir=out_dir
                )
                print(f"idx: {idx}")
                print(f"input: {test_audio_transcripts[step]}")
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")

        print("*********************** test end *****************************")



if __name__ == "__main__":
    test_infer()
