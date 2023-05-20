import streamlit as st
import tempfile
import os

from google.oauth2.service_account import Credentials
from google.cloud import vision
from google.cloud import translate_v2 as translate
from PIL import Image
import pdfplumber

from langchain.text_splitter import TokenTextSplitter

import openai


# Create credentials from our GCP secrets
creds = Credentials.from_service_account_info(st.secrets["gcp"])
client = vision.ImageAnnotatorClient(credentials=creds)
translate_client = translate.Client(credentials=creds)


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

def process_and_convert(files):
    converted_text = ""

    for file in files:
        content = file.read()

        if file.type.startswith('image/'):
            try:
                # Open the image file with PIL
                image = Image.open(file)

                # Convert HEIC/HEIF to JPEG if needed
                if image.format in ["HEIC", "HEIF"]:
                    with tempfile.NamedTemporaryFile(suffix='.jpeg') as temp_image:
                        image = image.convert("RGB")
                        image.save(temp_image.name, format="JPEG")
                        temp_image.seek(0)
                        content = temp_image.read()

                # Pass the image content to Google Cloud Vision API
                image = vision.Image(content=content)
                response = client.text_detection(image=image)
                texts = response.text_annotations
                converted_text += texts[0].description if texts else "No text found in the image."

            except Exception as e:
                converted_text += f"Error processing image: {str(e)}"

        elif file.type == 'application/pdf':
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
                    temp_pdf.write(content)
                    temp_pdf.seek(0)
                    with pdfplumber.open(temp_pdf.name) as pdf:
                        for page in pdf.pages:
                            converted_text += page.extract_text() + "\n"
                            
            except Exception as e:
                converted_text += f"Error processing PDF file: {str(e)}"

        else:
            converted_text += "Unsupported file format. Please upload an image or PDF file."

    return converted_text

def parse(raw_text):
    # raw_text를 split하기
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=30)
    texts = text_splitter.split_text(raw_text) # list[strings]
    print(len(texts))
    return texts

    # summarize chain 생성 및 실행 (prompt 길이 증가율이 양수)
    # 지금으로선 문서 길이가 너무 길어서 summary가 약 6000토큰 넘어가면 overflow되는 문제 있음
    # 또 다른 방법: chat history 사용하기? (scope 설정하면 비용증가 조절 가능할듯)
    
chat_history = [
        {"role": "system", 
         "content": """
        Your job is to summarize the given part of the article used as reading material in a university for the student.
        The summary should include all the important details and key ideas.
        Your summary should start by completing the former summary.
        The summary should be in a well-organized markdown style, using headings and bullet points."""}
    ]     

def summarize(chunk):
    new_message = {"role": "user", "content": "{text}".format(text=chunk)}
    chat_history.append(new_message)

    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=chat_history,
    temperature=0.3
    )
    resp = response['choices'][0]['message']

    chat_history.append(resp)
    summary = resp['content']

    return summary



def translate_text(target, text):
    try:
        result = translate_client.translate(text, target_language=target)

        return result['translatedText']
    except Exception as e:
        print(f"An error occurred during translation: {e}")

def main():
    st.title("Class Material Summarizer")
    st.write("Upload your class handouts and get a summary!")

    uploaded_files = st.file_uploader("Upload image or PDF files", accept_multiple_files=True)
    convert_button = st.button("Convert Files")

    if uploaded_files and convert_button:
        st.session_state['converted_text'] = process_and_convert(uploaded_files)

        st.header("Converted Text")
        st.write(st.session_state['converted_text'])

    # Button to generate summary
    if st.button("Generate Summary"):
        if 'converted_text' in st.session_state:
            st.session_state['parsed_text'] = parse(st.session_state['converted_text']) # list[str]

            # Generate the summary
            st.header("Summary")
            st.session_state['summarized_text'] = ''

            image_placeholder = st.empty()
            image_placeholder.image('breakdance.gif')
            text_placeholder = st.empty()
            text_placeholder.markdown("요약하는 중...")

            for chunk in st.session_state['parsed_text']:
                summary = summarize(chunk)
                print(summary)
                st.write(summary)
                st.session_state['summarized_text'] += summary

            image_placeholder.empty()
            text_placeholder.empty()
        else:
            st.error("No text to summarize. Please upload a file and convert it first.")

    # Check if summarized text is in the session state
    #if 'summarized_text' in st.session_state:
        #if st.checkbox("Translate Summary to Korean", value=False):
            # when the toggle is checked, translate and display the translated text
            # Avoid repeated translations by checking if it's already done
            
            #if 'translated_text' not in st.session_state:
                #st.session_state['translated_text'] = translate_text('ko', st.session_state['summarized_text'])
            #st.header("Translated Summary")
            #st.write(st.session_state['translated_text'])

        #else:
            # when the toggle is unchecked, display the original text
            #st.header("Original Summary")
            #st.write(st.session_state['summarized_text'])

    

    # Button to generate questions
    if st.button("Generate Questions"):
            # Question generation using GPT-3.5 API
            # Add your code here to make a request to the GPT-3.5 API for question generation

            # Display the generated questions
            st.header("Questions")
            # for question in questions:
                # st.write(question)
        
    if st.button("Chat"):
            # Chat feature using GPT-3.5 API

            st.header("Chat")

if __name__ == "__main__":
    main()
