import streamlit as st
from google.cloud import storage
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
import pytube
import unidecode
import moviepy.editor as mpe


# Streamlit configuration
st.title("YouTube Video Summarization and Chapter Index")

# Vertex AI Setup (Move this to a secure location if deploying)
PROJECT_ID = "your-project-id"  
LOCATION = "us-central1" 
BUCKET_NAME = "your-bucket-name"
BUCKET_URI = f"gs://{BUCKET_NAME}"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-pro-preview-0409")

# Helper Functions
def generate_response(prompt, video_part):
    params = {
        "safety_settings": {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        "generation_config": {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95
        },
        "stream": True
    }
    
    video_part = Part.from_uri(uri=video_part, mime_type="video/mp4")
    responses = model.generate_content([prompt, video_part], **params)
    return responses

def download_and_upload_video(video_url):
    yt = YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    filename = stream.download()
    uploaded_video = upload_file_to_bucket(BUCKET_NAME, filename)
    return uploaded_video

def upload_file_to_bucket(bucket_name, source_file_path_name):
    destination_blob_name = unidecode.unidecode(source_file_path_name.replace(' ', '_'))
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path_name)
    return f"gs://{bucket_name}/{destination_blob_name}"


# Streamlit App Logic
video_url = st.text_input("Enter YouTube Video URL:")
if st.button("Process Video"):
    if video_url:
        with st.spinner("Processing video..."):
            uploaded_video = download_and_upload_video(video_url)
            video = mpe.VideoFileClip(uploaded_video) 

            # Splitting into 5-minute chunks
            duration = video.duration
            chunk_size = 300
            num_chunks = int(duration / chunk_size) + 1
            video_parts = [f"{uploaded_video}#t={i*chunk_size},{min((i+1)*chunk_size, duration)}" for i in range(num_chunks)]

            prompt = """Create a chapter index for this video and generate a nice HTML code to present the chapter index."""
            responses = [generate_response(prompt, part) for part in video_parts]

            result = []
            for response_list in responses:
                for response in response_list:
                    result.append(response.text)

            st.markdown("\n".join(result))  # Display in markdown for HTML formatting
    else:
        st.warning("Please enter a YouTube video URL.")
