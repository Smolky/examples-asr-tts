from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
import sys
import scipy.io.wavfile


models, cfg, task = load_model_ensemble_and_task_from_hf_hub ("facebook/tts_transformer-es-css10", arg_overrides={"vocoder": "hifigan", "fp16": False})


model = None
for my_model in models:
	model = my_model

TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator (models, cfg)

text = "Muchísimas gracias por reservar. ¿Confirmamos entonces que la cita es a las 9 de la mañana?."

sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction (task, model, generator, sample)

ipd.Audio(wav, rate=rate)

print (wav.numpy ())
print (rate)

scipy.io.wavfile.write("output.wav", rate, wav.numpy ())


