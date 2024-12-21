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
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper

from moshi.models import loaders

torch.set_printoptions(sci_mode=False)

NUM_CODEBOOKS = 8

# TODO
text_vocabsize = 64000
text_specialtokens = 64
audio_vocabsize = 2048
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr_t = text_vocabsize + 4
_tts_t = text_vocabsize + 5

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_asr_a = audio_vocabsize + 4
_tts_a = audio_vocabsize + 5



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
    input_ids_item = [[] for _ in range(NUM_CODEBOOKS+1)]
    text_tokens = text_tokenizer.encode(text)
    for i in range(8):
        input_ids_item[i] = [layershift(_input_a, i)] + [layershift(_pad_a, i)] * len(text_tokens) + [layershift(_eoa, i), layershift(_tts_a, i)]
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


def get_input_ids_mimi(
    audio_wav, mimi_model, device, latency_list,
    special_token_a=_answer_a, special_token_t=_answer_t, text=None, text_tokenizer=None
):

    if text and text_tokenizer:
        text_tokens = text_tokenizer.encode(text).tolist()
    else:
        text_tokens = None
    
    with torch.no_grad():
        audio_wav = torch.from_numpy(audio_wav).unsqueeze(0).unsqueeze(0).to(device)
        audio_tokens = mimi_model.encode(audio_wav)[0]

    T = audio_tokens.size(-1)
    input_ids = []
    for i in range(8):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item +=  [layershift(_pad_a, i)] * latency_list[i] + [layershift(x, i) for x in audio_tokens[i]]
        # input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
        input_ids.append(torch.tensor(input_ids_item[:1+T]).unsqueeze(0))
    
    if text_tokens:
        assert len(text_tokens) <= T
        input_id_T = torch.tensor([_input_t] +  text_tokens + [_pad_t] * (T-len(text_tokens)-1) + [_input_t]) 
    else:
        input_id_T = torch.tensor([_input_t] + [_pad_t] * (T-1) + [_input_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return input_ids


def get_input_ids_asr(
    audio_wav, mimi_model, device, latency_list
):
    with torch.no_grad():
        audio_wav = torch.from_numpy(audio_wav).unsqueeze(0).unsqueeze(0).to(device)
        audio_tokens = mimi_model.encode(audio_wav)[0]

    T = audio_tokens.size(-1)
    input_ids = []
    for i in range(8):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * latency_list[i] + [layershift(x, i) for x in audio_tokens[i]] +  [layershift(_eoa, i)] + [layershift(_pad_a, i)] * (1-latency_list[i]) + [layershift(_asr_a, i)]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    

    input_id_T = torch.tensor([_input_t] + [_pad_t] * (T+1)  + [_eot, _asr_t])

    input_ids.append(input_id_T.unsqueeze(0))
    return input_ids



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
    return audio


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


def A1_A2(fabric, input_ids, leng, model, text_tokenizer, step,
          mimi_model, out_dir=None):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_AA(
        model,
        None,
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
        layershift_stride=padded_audio_vocabsize,
        moshi_infer=True,
    )
    
    audiolist = tokenlist[:NUM_CODEBOOKS]
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
    
    with torch.inference_mode(), mimi_model.streaming(1):
        codecs = torch.tensor(audiolist).unsqueeze(0).cuda()
        codecs = codecs[:, :, :-1]
        codecs = torch.where(codecs >= 2048, torch.tensor(0), codecs)
        if codecs.size(-1) == 0:
            print("audio codecs is 0")  
        else:
            audio_hat = mimi_model.decode(codecs)
            sf.write(
                f"{out_dir}/{step:02d}.wav",
                audio_hat.squeeze().cpu().numpy(),
                24000,
            )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def A1_T1(fabric, input_ids, model, text_tokenizer, step):
    # with fabric.init_tensor():
    #     model.set_kv_cache(batch_size=1)
    tokenlist = generate_ASR(
        model,
        None,
        input_ids,
        [0],
        ["A1T1"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_pad_a,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        layershift_shift=padded_text_vocabsize,
        layershift_stride=padded_audio_vocabsize,
        moshi_infer=True,
        num_codebooks=NUM_CODEBOOKS,
        include_prompt=True,
        generate_text=True,
    )
    # model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_A2(fabric, input_ids, model, text_tokenizer, step,
          mimi_model, out_dir=None):
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
        layershift_stride=padded_audio_vocabsize,
        moshi_infer=True,
        num_codebooks=NUM_CODEBOOKS
    )

    audiolist = tokenlist[:NUM_CODEBOOKS]
    tokenlist = tokenlist[-1]

    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    if out_dir is None:
        out_dir = "./output/default/T1-A2"
    else:
        out_dir = out_dir + "/T1-A2"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.inference_mode(), mimi_model.streaming(1):
        audiolist = [x[:-1] if idx == 0 else x[1:] for idx, x in enumerate(audiolist)]
        codecs = torch.tensor(audiolist).unsqueeze(0).cuda()
        # codecs = codecs[:, :, :-1]
        print(f"audio codecs >= 2048, number: {torch.sum(codecs >= 2048).item()}") 
        codecs = torch.where(codecs >= 2048, torch.tensor(0), codecs)
        if codecs.size(-1) == 0:
            print("audio codecs is 0")  
        else:
            audio_hat = mimi_model.decode(codecs)
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
    mimi_weight = "/lp/models/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = loaders.get_mimi(mimi_weight, device='cuda')
    mimi.set_num_codebooks(8)  
    mimi.cuda()

    text_tokenizer = Tokenizer("/lp/models/Yi-6B")

    if True:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            device_map="cpu",
            torch_dtype=torch.float,
            trust_remote_code=True
        )
        print(f"load model from ckpt_dir {ckpt_dir}")

    model.to(device).eval()

    return None, model, text_tokenizer, mimi


def test_infer():
    device = "cuda:0"
    out_dir = f"./output/{get_time_str()}"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_tllmall_librilight_quora_zhihu/checkpoint/iter_0012000_hf_A"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_mb2_tts8_fdecoder_librilight_quora_zhihu/checkpoint/iter_0022000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_mb2_tts8_fllmall_librilight_quora_zhihu/checkpoint/iter_0026000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_tllmall_librilight_quora_zhihu/checkpoint/iter_0019500_hf"

    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_tts/checkpoint/iter_0004000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_tts_asr_ntp/checkpoint/iter_0004000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_asr/checkpoint/iter_0003000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_librilight_quora_zhihu_yunting_spotify_asr/checkpoint/iter_0001000"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_librilight_quora_zhihu_yunting_spotify_asr/checkpoint/iter_0001000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_tts_asr_ntp/checkpoint/iter_0007000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_librilight_quora_zhihu_yunting_spotify_asr/checkpoint/iter_0004000_hf"
    # ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_tts/checkpoint/iter_0008000_hf"
    ckpt_dir = "/lp/code/mla/MLA_Megatron-LM/out/mimi_pretrain/yi6b_bs1k_tts8_fdecoder_librilight_quora_zhihu_yunting_spotify_tts/checkpoint/iter_0009500_hf"
    
    # if not os.path.exists(ckpt_dir):
    #     print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
    #     download_model(ckpt_dir)

    fabric, model, text_tokenizer, mimi_model = load_model(ckpt_dir, device)

    # task = ['A1A2', 'asr', "T1A2", "AA-BATCH", 'T1T2', 'AT']
    # task = ["AA-BATCH"]
    # task = ["AT"]
    # task = ["A1A2"]
    # task = ['T1A2']
    # task = ["asr", "tts"]
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
        "Your mum is doing for you what she thinks is best.Unfortunately, like all parents, it's hard when to know you stop and your child begins.As you get older you will change regardless and she will feel she has lost control.Comfort her by saying you will never leave, your heart is always with her wherever you go.That God has chosen your life and it's not the same as what she thinks.That what works for her, doesn't work for everyone else, as everyone is different.Reassure her she is doing a great job.",
        "Map out your hours!Everyone is given the same twenty four hours in a day but the key to not getting burnt out is to plan ahead.Make rough to do lists for the day and goals for the week",
        "As a student, i know it's very difficult to balance personal or social life with studies as ca is a very long journey with vast syllabus. The quantum of syllabus makes us dull to enjoy other things in life.Like everytime you get a chance to enjoy, you always think about completion of syllabus and time is very limited but that make us monotonous. Monotonous work make us inefficient and inefficiency limits our scope of achieving targets and when the targets not get finished, we start doubting self.",
        "面试、重大场合的时候,让自己闪亮一下,又何乐而不为呢。这比长高可简单多了。不化妆的女生其实也很美。自然清纯,悠然自得有些女生化妆反而会显得老态,难以保持清纯的样子,不化妆显得更自然,更美。不化妆不代表不对皮肤进行护理,化妆对皮肤的负担很重",
        "之前由于院系大调整从浙大拆了出去,这些高校从弱小到壮大,浙江大学后来又将这些合并成一个巨无霸高校,就有人调侃是其是坐享其成。吐槽点:各种各样武汉大学和某些明星一样,天生招黑体质,不管做什么都有人黑。至于原因,有的说它是发展太快",
        "To improve the TRUE economy, which is output of useful products & services, we need to shift people from low to negative productivity jobs in government to high productivity jobs in the private sector!",
        "Among the interviews they’ve been conducting, law enforcement interviewed a female employee at the hostel who said, at one point, she asked the then masked man to lower his mask while flirting with him — which is when the photos released by the New York Police Department today were captured, the official said.",
        "上周末，一篇 Google DeepMind 的论文引发了 AI 圈的关注。研究者引入了「苏格拉底式学习」，这是 AI 中递归自我完善的一种新方法。这种方法使系统能够自主增强其能力，超越初始训练数据的限制。通过利用结构化的「语言游戏」，该技术可以为实现通用人工智能提供了实用的路线图。",
    ]

    latency_list = [0]+ [1] * 7

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
                    audio_wav = load_audio(path)
                    # input_ids = get_input_ids_mimi(
                    #     audio_wav, mimi_model, device, latency_list, text=test_audio_transcripts[idx], text_tokenizer=text_tokenizer
                    # )
                    input_ids = get_input_ids_mimi(
                        audio_wav, mimi_model, device, latency_list, text=None, text_tokenizer=text_tokenizer
                    )
                    leng = input_ids[0].size(-1)
                    text = A1_A2(
                        fabric,
                        input_ids,
                        leng,
                        model,
                        text_tokenizer,
                        step,
                        mimi_model,
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
                audio_wav = load_audio(path)
                # audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_pad_a, special_token_t=_answer_t)
                input_ids = get_input_ids_asr(audio_wav, mimi_model, device, latency_list)
                output = A1_T1(fabric, input_ids, model, text_tokenizer, index).lower().replace(',','').replace('.','').replace('?','')
                print(f"audio_path: {path}")
                print(f"audio transcript: {test_audio_transcripts[index]}")
                print(f"asr output: {output}")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                index += 1

        if "tts" in task:
            step = 0
            print("\n")
            print("===============================================================")
            print("                       testing TTS")
            print("===============================================================")
            for idx, text in enumerate(tts_text_list):
                input_ids = get_input_ids_TTS(text, text_tokenizer)
                text_output = T1_A2(fabric, input_ids, model, text_tokenizer, step,
                                    mimi_model, out_dir=out_dir)
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
