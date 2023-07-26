import streamlit as st
import torch
import sentencepiece
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


st.write("""
# summerize your text
"""
)

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

text_input = st.text_area("Text to summarizer")

if text_input:
    tokenized_text=tokenizer.encode_plus(str(text_input),padding="longest",truncation=True, return_tensors="pt")

    summary = model.generate(**tokenized_text)

    summary_text =  [ tokenizer.decode(summary[0],skip_special_tokens=True) ]
                     
    # summary_text = [ (summary_text.replace("<pad>", " ").replace("</s>", " ") ]
 
    st.write("##Summarized text")
    st.write(" ".join(summary_text))
