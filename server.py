import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
# from inference import OmniInference
from inference_yi_omni import OmniInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False
        self.client = OmniInference(ckpt_dir, device)
        self.client.warm_up()

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def chat(self) -> Response:

        req_data = flask.request.get_json()
        try:
            data_buf = req_data["audio"].encode("utf-8")
            data_buf = base64.b64decode(data_buf)
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                audio_generator = self.client.run_AA_stream(f.name, stream_stride, max_tokens)
                return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            print(traceback.format_exc())


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False, ckpt_dir="/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_bs512_4aatmode_d1030_load_tts8_ckpt_train/checkpoint/iter_0015000_hf")
    return server.server


def serve(ip='0.0.0.0', port=60808):

    OmniChatServer(ip, port=port, run_app=True, ckpt_dir="/lp/code/mla/MLA_Megatron-LM/out/test_audio_instruct/yi6b_bs512_4aatmode_d1030_load_tts8_ckpt_train/checkpoint/iter_0015000_hf")


if __name__ == "__main__":
    import fire
    fire.Fire(serve)
    
