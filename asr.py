from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel ("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
audio_paths = ["./sample.wav"]

transcriptions = model.transcribe(audio_paths)

print (transcriptions)
