import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize the pipelines and clients
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

llm = OpenAI(temperature=1)
prompt_template_name = PromptTemplate(
    input_variables=['scenario'],
    template="""
You are a story teller:
You can generate a short story based on a simple narrative, the story should be no more than 50 words;

CONTEXT: {scenario}
STORY:
"""
)

story_llm = LLMChain(llm=llm, prompt=prompt_template_name)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_file(text: str) -> str:
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",  # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    return save_file_path

# Streamlit frontend
st.title("Image to Story to Audio App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image to Text
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    caption = image_to_text(image_path)[0]["generated_text"]
    st.write(f"Generated Caption: {caption}")

    # Text to Story
    story = story_llm.run(caption)
    st.write(f"Generated Story: {story}")

    # Text to Speech
    audio_file_path = text_to_speech_file(story)
    st.write("Audio file has been generated!")
    
    # Display audio file
    st.audio(audio_file_path, format="audio/mp3")
    
    # Provide download link
    st.download_button(
        label="Download Audio",
        data=open(audio_file_path, "rb").read(),
        file_name=audio_file_path,
        mime="audio/mp3"
    )
    
    # Clean up temporary files
    os.remove(image_path)
    os.remove(audio_file_path)
