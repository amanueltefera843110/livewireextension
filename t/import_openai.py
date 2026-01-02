from openai import OpenAI
import pyttsx3

# Initialize text-to-speech engine with error handling
try:
    tts_engine = pyttsx3.init()
    # Optional: Configure voice settings
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Text-to-speech not available: {e}")
    TTS_AVAILABLE = False

# Add your complete API key here
client = OpenAI(api_key="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")

def speak(text):
    """
    Convert text to speech
    """
    if TTS_AVAILABLE:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    else:
        print("(Speech output disabled)")

def chat_with_gpt(prompt, conversation_history=None):
    """
    Send a message to ChatGPT and get a response
    """
    if conversation_history is None:
        conversation_history = []
    
    # Add the user's message to the conversation
    conversation_history.append({"role": "user", "content": prompt})
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )
        
        # Get the assistant's reply
        assistant_message = response.choices[0].message.content
        
        # Add assistant's response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message, conversation_history
    
    except Exception as e:
        error_msg = f"Error communicating with OpenAI: {e}"
        print(error_msg)
        return error_msg, conversation_history

def main():
    """
    Main function to run the chatbot with speech
    """
    print("ChatGPT Speaking Chatbot - Type 'quit' to exit\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'bye']:
            goodbye_msg = "Goodbye!"
            print(goodbye_msg)
            speak(goodbye_msg)
            break
        
        # Get response from ChatGPT
        response, conversation_history = chat_with_gpt(user_input, conversation_history)
        
        # Print and speak the response
        print(f"\nChatGPT: {response}\n")
        speak(response)

if __name__ == "__main__":
    main()