import streamlit as st
import tempfile

from google.oauth2.service_account import Credentials
from google.cloud import vision
from google.cloud import translate_v2 as translate
from PIL import Image
import io
import pdfplumber

from langchain.text_splitter import TokenTextSplitter

import openai

import time

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
                # Reset the pointer to the start of the file
                file.seek(0)

                # Open the file directly with pdfplumber
                with pdfplumber.open(file) as pdf:
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

usage = 'article'
usage_prompt = "the ppt slides for university students. You should Minimize prose." if usage == 'ppt' else "an article used as reading material in a university for the student."


chat_history = [
        {"role": "system", 
         "content": """
        Your job is to summarize the given part of {usage}
        The summary should include all the important details and key ideas.
        If needed, complete the unfinished part of the former summary.
        BE PRECISE, and do not rewrite the whole summary each time.
        The summary should be in the language of the original text.
        The structure should be coherent with the former summaries.
        The summary should be in a well-structured markdown style, using headings, subheadings and bullet points.
        """.format(usage=usage_prompt)}
    ]


def translate_text(target, text):
    try:
        result = translate_client.translate(text, target_language=target)

        return result['translatedText']
    except Exception as e:
        print(f"An error occurred during translation: {e}")

def main():
    st.title("썸머라이저")
    st.write("영어 자료를 업로드하고 요약을 받으세요.")

    uploaded_files = st.file_uploader("이미지 또는 Pdf 파일 업로드", accept_multiple_files=True)
    convert_button = st.button("변환하기")



    if uploaded_files and convert_button:
        st.header("변환 결과")

        image_placeholder = st.empty()
        image_placeholder.image('breakdance.gif')
        text_placeholder = st.empty()
        text_placeholder.markdown("변환하는 중...")

        st.session_state['converted_text'] = process_and_convert(uploaded_files)
        st.write(st.session_state['converted_text'])

        image_placeholder.empty()
        text_placeholder.empty()

    if 'converted_text' in st.session_state:
        # Get the text
        text = st.session_state['converted_text']

        # Create a byte stream of the text
        text_bytes = text.encode()

        # Create a download button for the byte stream
        st.download_button(
            label="Download text file",
            data=io.BytesIO(text_bytes),
            file_name="converted_text.txt",
            mime="text/plain",
        )

    # Button to generate summary
    if st.button("요약 생성하기"):
        if 'converted_text' in st.session_state:
            st.session_state['parsed_texts'] = parse(st.session_state['converted_text']) # list[str]

            # Generate the summary
            st.header("Summary")
            st.session_state['summarized_text'] = ''

            image_placeholder = st.empty()
            text_placeholder = st.empty()
            image_placeholder.image('breakdance.gif')
            text_placeholder.markdown("요약하는 중...")

            # streaming

            fin_res = ''
            for i, chunk in enumerate(st.session_state['parsed_texts']):

                # 마지막 chunk일 경우를 제외하고 결론을 내리지 않게 설정
                if i == len(st.session_state['parsed_texts']) -1 :
                    conclusion = ""
                else:
                    # context token 수 길어지는거 방지
                    if 0 < i < 4:
                        del chat_history[-2] # user message(본문이라 졸라김)만 삭제
                    elif i >= 4:
                        del chat_history[1] # 남아있는 요약 중 첫번째 삭제
                        del chat_history[-2] # 남아있는 user message 중 첫번째 삭제

                        # i = 0: [{system}] -> 안삭제 -> [{system}, {user}, {assistant}]
                        # i = 1: [{system}, {user}, {assistant}] -> [{system}, {assistant}] -> [{system}, {assistant}, {user}, {assistant}]
                        # i = 2: [{system}, {assistant}, {user}, {assistant}] -> [{system}, {assistant}, {assistant}] -> [{system}, {assistant}, {assistant}, {user}, {assistant}]
                        # i = 3: [{system}, {assistant}, {assistant}, {user}, {assistant}] -> [{system}, {assistant}, {assistant}, {assistant}] -> [{system}, {assistant}, {assistant}, {assistant}, {user}, {assistant}]
                        # i = 4: [{system}, {assistant1}, {assistant2}, {assistant3}, {user4}, {assistant4}] -> [{system}, {assistant2}, {assistant3}, {assistant4}] -> [{system}, {assistant2}, {assistant3}, {assistant4}, {user5}, {assistant5}]
                    
                    conclusion = " (DO NOT Rewrite the whole summary, and DO NOT MAKE CONCLUSION yet - there's more text to come!)"

                
                print('--------')
                
                res_box = st.empty()
                new_message = {"role": "user", "content": "{text}".format(text=chunk+conclusion)}
                chat_history.append(new_message)
                report = []
                result = ''

                # switching 방식 바꿈
                if i <= 1:
                    response = openai.ChatCompletion.create(
                    model= "gpt-4",
                    messages=chat_history,
                    temperature=0.3,
                    stream=True
                    )
                else:
                    response = openai.ChatCompletion.create(
                    model= "gpt-3.5-turbo",
                    messages=chat_history,
                    temperature=0.3,
                    stream=True
                    )
                
                image_placeholder.empty()
                text_placeholder.empty()

                print(chat_history)
                
                for resp in response:
                    try:
                        report.append(resp['choices'][0]['delta']['content'])
                    except KeyError:
                        report.append(' ')
                    result = "".join(report)
                    res_box.markdown(result)

                chat_history.append({"role": "assistant", "content": result})
                print("결과: " + result)
                fin_res += result
                res_box.markdown(result)

                image2_placeholder = st.empty()
                text2_placeholder = st.empty()
                image2_placeholder.image('breakdance.gif')
                text2_placeholder.markdown('다음 장 읽어보는 중...')

                time.sleep(15) # Rate Limit 방지

                image2_placeholder.empty()
                text2_placeholder.empty()


            st.session_state['converted_text'] = fin_res

        else:
            st.error("No text to summarize. Please upload a file and convert it first.")

    # Check if summarized text is in the session state
    # if 'summarized_text' in st.session_state:
        # if st.checkbox("Translate Summary to Korean", value=False):
            # when the toggle is checked, translate and display the translated text
            # Avoid repeated translations by checking if it's already done
            
            # if 'translated_text' not in st.session_state:
                # st.session_state['translated_text'] = translate_text('ko', st.session_state['summarized_text'])
            # st.header("Translated Summary")
            # st.write(st.session_state['translated_text'])



    # Button to generate questions
    #if st.button("Generate Questions"):
            # Question generation using GPT-3.5 API
            # Add your code here to make a request to the GPT-3.5 API for question generation

            # Display the generated questions
            #st.header("Questions")
            # for question in questions:
                # st.write(question)
        
    #if st.button("Chat"):
            # Chat feature using GPT-3.5 API

            #st.header("Chat")

if __name__ == "__main__":
    main()
