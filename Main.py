import streamlit as st
import utils


from gensim.parsing.preprocessing import preprocess_string

def main():
    st.header("Understand the Topic of Different Legal Text Clauses")
    st.divider()

    st.subheader("Step 1: Choose a sample text to analyze: \n Sample txt")
    df_file = utils.upload_file("Upload text file")

    if df_file is not None:
        # Display the processed text
        words = df_file.split()[:300]
        st.write("Sample Case Text:")
        st.write(' '.join(words))

    st.subheader("Step 2: Choose a model from the left sidebar")
    model_name = utils.sidebar()

    if model_name:
        st.write(f"Selected model: {model_name}")

    # Perform classification based on the selected model
    if model_name == "Sentiment Analysis BERT":
        sentiment_score = utils.analyze_sentiment(df_file)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Sentiment Score: {sentiment_score}")

    elif model_name == "LDA Model":
        topic_distribution = utils.perform_lda(df_file)
        st.subheader("LDA Model Result:")
        st.write("Topic Distribution:")
        st.write(topic_distribution)

if __name__ == "__main__":
    main()