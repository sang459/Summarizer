import streamlit as st
import tempfile
import os

from google.oauth2.service_account import Credentials
from google.cloud import vision
from PIL import Image
from pdfreader import SimplePDFViewer

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


# Create credentials from our GCP secrets
creds = Credentials.from_service_account_info(st.secrets["gcp"])
client = vision.ImageAnnotatorClient(credentials=creds)

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
            viewer = SimplePDFViewer(file)
            viewer.render()
            num_pages = len(viewer.pages)

            for page in range(num_pages):
                page_obj = pdf_reader.getPage(page)
                converted_text += page_obj.extractText()

        else:
            converted_text += "Unsupported file format. Please upload an image or PDF file."

    return converted_text

def summary(raw_text):
    # raw_text를 split하기
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=30)
    texts = text_splitter.split_text(raw_text)
    print(len(texts))
    print(texts[1])

    # document 생성
    docs = [Document(page_content=t) for t in texts]

    # summarize chain 생성 및 실행 (prompt 길이 증가율이 양수)
    # 지금으로선 문서 길이가 너무 길어서 summary가 약 6000토큰 넘어가면 overflow되는 문제 있음
    # 또 다른 방법: chat history 사용하기? (scope 설정하면 비용증가 조절 가능할듯)
    prompt_template = """FULL TEXT:


    {text}


    DETAILED SUMMARY (using markdown headings and bullet points):"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a summary of the article.\n"
        "This is the existing summary of the article up to a certain point for your context: {existing_answer}\n"
        "The next part of the article is as follows.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Produce a detailed summary of the given part of the article, using markdown headings and bullet points.\n" # detailed가 중요한듯!
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    # 한 chunk씩 썸머리하고 합쳐서 반환
    chain = load_summarize_chain(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="refine", 
                                 return_intermediate_steps=True, question_prompt=PROMPT, 
                                 refine_prompt=refine_prompt)
    summarizations = chain({"input_documents": docs}, return_only_outputs=True)['intermediate_steps']
    summarized_text = "\n".join(summarizations)

    return summarized_text

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
        # Check if converted text is in the session state
        if 'converted_text' in st.session_state:
            # Summarization using GPT-3.5 API
            summarized_text = summary(st.session_state['converted_text'])
            
            # Display the generated summary
            st.header("Summary")
            st.write(summarized_text)
        else:
            st.error("No text to summarize. Please upload a file and convert it first.")
    







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
