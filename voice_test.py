import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set speech properties
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Select voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Test voice output
print("Testing VisionAssist voice...")

engine.say("Hello! Welcome to Vision Assist.")
engine.say("Object detection is working.")
engine.say("I can see a person on your right.")
engine.say("Please be careful. Obstacle ahead.")

# Speak all messages
engine.runAndWait()

print("Voice test completed successfully!")