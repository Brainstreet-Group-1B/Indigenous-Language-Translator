import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSeq2SeqLM.from_pretrained("./model")

st.title("English to Luganda Translator")
st.write("Translate text from English to Luganda!")

input_text = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if input_text:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Translated Text:")
        st.success(translated_text)
    else:
        st.warning("Please enter some text for translation.")
