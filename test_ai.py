import audiopy as ap
import os

print("🤖 Starting AI Features Demo...")

# 1. Generate realistic speech using Text-To-Speech
print("\n[1/4] Generating speech using Google TTS...")
text = "I am so incredibly happy and excited to be working with you today! This is amazing!"
print(f"      Text: \"{text}\"")

# This returns a numpy array and sample rate
y_speech, sr = ap.speak(text)
ap.save("ai_speech_sample.wav", y_speech, sr)
print("      Saved generated speech to 'ai_speech_sample.wav'")

# 2. Test Audio Classification
print("\n[2/4] Running Audio Classification (AST model)...")
print("      (First run will download the model...)")
classification_results = ap.classify(y_speech, sr)
print(f"      Top Prediction: {classification_results[0]['label']} ({classification_results[0]['score']*100:.1f}%)")

# 3. Test Speech Transcription
print("\n[3/4] Running Speech Transcription (Whisper model)...")
print("      (First run will download the model...)")
transcription = ap.transcribe(y_speech, sr)
print(f"      Transcription: \"{transcription}\"")

# 4. Test Emotion Detection
print("\n[4/4] Running Emotion Detection (Wav2Vec2 model)...")
print("      (First run will download the model...)")
emotion_results = ap.detect_emotion(y_speech, sr)
print(f"      Dominant Emotion: {ap.dominant_emotion(y_speech, sr)}")
print("      All scores:")
for result in emotion_results:
    print(f"        - {result['label']}: {result['score']*100:.1f}%")

print("\n✅ AI Demo completed successfully!")
