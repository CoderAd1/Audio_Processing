import streamlit as st
from fpdf import FPDF

# Function to convert text to PDF
from openai import OpenAI
import io
import time
import streamlit as st
import pandas as pd
from fpdf import FPDF
# Function to process audio
client = OpenAI()
prompt_config={"Extraction":
               "You are an api that inputs a list of words and its start and end time stamps \
            and outputs a python dictonary with key symptoms\
            and values are symptoms extracted from the text \
            given by the user along with the time stamp and another key disease and value\
            the disease from the text along with the time stamp and another key recommendation \
            from the doctor with value of recommendation given by the \
            doctor in 3 or 4 words along with the time stamp, dont use ``` or python.Finally \
            Output should be in this format {'symptoms': [{'symptom1': (start timestamp,end timestamp)}, {'symptom2': (start timestamp,end timestamp)}],\
 'disease': {'diesase1': (start timestamp,end timestamp)},\
 'recommendation': {'recommendation1': (start timestamp,end timestamp)}}\
            ",
               "Segmentation":
               "You are an api that inputs a list of sentences and its time stamps \
            you have to identift which sentence was spoken by the doctor and which by the  \
            patient and mark and return the marked text in the given format  in new lines \
            Doctor : doctors texts \
            Patient: patiensts texts\
            and nothing else\
            "}
openai_config={"temperature":1,
               "max_tokens":256,
               "top_p":1,
               "frequency_penalty":0,
               "presence_penalty":0}
def exctractor(l):
    prompts=[
        {
        "role": "system",
        "content": prompt_config["Extraction"]
        },
        {
        "role": "user",
        "content": str(l)
        }]
    t=time.time()
    response2 = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=prompts,
        temperature=openai_config["temperature"],
        max_tokens=openai_config["max_tokens"],
        top_p=openai_config["top_p"],
        frequency_penalty=openai_config["frequency_penalty"],
        presence_penalty=openai_config["presence_penalty"]
        )
    #print(time.time()-t)
    return eval(response2.choices[0].message.content)
def Segmentor(l):
    prompts=[
        {
        "role": "system",
        "content": prompt_config["Segmentation"]
        },
        {
        "role": "user",
        "content": str(l)
        }]
    t=time.time()
    response2 = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=prompts,
        temperature=openai_config["temperature"],
        max_tokens=openai_config["max_tokens"],
        top_p=openai_config["top_p"],
        frequency_penalty=openai_config["frequency_penalty"],
        presence_penalty=openai_config["presence_penalty"]
        )
    #print(time.time()-t)
    return response2.choices[0].message.content
def transcribe_audio(audio_file):
    t=time.time()
    transcription = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file,
        response_format="verbose_json",
    )
    print("transcription time: ",time.time()-t)
    # print(transcription.text)
    t=time.time()
    l=[]
    for i in transcription.segments:
        l.append(
            (i["text"],
             round(i["start"],2),round(i["end"],2)))
    print("Preprocessing time: ",time.time()-t)
    t=time.time()
    segmented=Segmentor(l)
    print("Segmentation time: ",time.time()-t)
    t=time.time()
    extracted=exctractor(l)
    print("Extraction time time: ",time.time()-t)
    return transcription.text,extracted,segmented
def text_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    return pdf

# Streamlit app
def main():
    st.title("Audio Processing")
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

    if audio_file is not None:
        # Read the contents of the file
        transcript,extract,segment = transcribe_audio(audio_file)
        st.subheader("Transcription:")
        st.write(transcript)
        st.subheader("Extracted Attributes")
        st.write(str(extract))
        st.subheader("Identified Speakers")
        st.write(segment)
        # Display the contents
        # st.subheader("Text from Uploaded File:")
        # st.text(file_contents)

        # Button to convert and download as PDF
        # if st.button("Convert to PDF"):
        #     pdf = text_to_pdf(str(extract))
            
            # Download the PDF file
        formatted_strings = ""
        data=extract
        # Extracting symptoms
        symptoms_str = 'symptoms : '
        for symptom_dict in data['symptoms']:
            for symptom, duration in symptom_dict.items():
                symptoms_str += f'{symptom}({duration[0]}, {duration[1]}), '
        symptoms_str = symptoms_str.rstrip(', ')
        formatted_strings += symptoms_str
        
        # Extracting disease
        disease_str = '\ndisease : '
        for disease, duration in data['disease'].items():
            disease_str += f'{disease}({duration[0]}, {duration[1]}), '
        disease_str = disease_str.rstrip(', ')
        formatted_strings += disease_str
        
        # Extracting recommendation
        recommendation_str = '\nrecommendation : '
        for recommendation, duration in data['recommendation'].items():
            recommendation_str += f'{recommendation}({duration[0]}, {duration[1]}), '
        recommendation_str = recommendation_str.rstrip(', ')
        formatted_strings += recommendation_str
        
        print(formatted_strings)
        
        pdf = text_to_pdf(formatted_strings)
        st.success("Data Extraction and Segmentation successful!")
        st.download_button(
            label="Download PDF",
            data=pdf.output(dest='S').encode('latin1'),
            file_name="Extracted_text.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
