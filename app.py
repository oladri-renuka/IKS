#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import streamlit as st
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from googletrans import Translator
from decimal import Decimal
import timeit

# Load the Vimanika Shastra dataset
vimanika_df = pd.read_csv("vimanika_shastra_dataset.csv")

# Load the Astro-Physics dataset
astro_df = pd.read_csv("astro-dataset.csv")

# Set logo image
logo_image = "ikslogo.png"

# Display logo and text in the sidebar
st.image(logo_image, width=160)
st.markdown("Indian Knowledge System")

# Title for the page
st.title("Multi-Domain Correlation App")

# User input for selecting domain
domain_options = ["---", "Vimanika Shastra", "Astro-Physics", "Vedic Maths"]
selected_domain = st.selectbox("Select Domain", domain_options, index=0)

if selected_domain == "Vimanika Shastra":
    st.title("Correlation between Vimanika Shastra and Modern Aerodynamic Concepts")
    # User input for Sanskrit shloka
    sanskrit_text = st.text_input("Enter the Sanskrit shloka:")

    if st.button("Find Correlations"):
        def translate_sanskrit_to_english(text):
            translator = Translator()
            translation = translator.translate(text, dest='en')
            return translation.text

        def process_vimanika_shastra(df, user_translation):
            def preprocess_text(text):
                text = text.lower()
                text = text.translate(str.maketrans("", "", string.punctuation))
                return text

            transliteration = transliterate(user_translation, sanscript.SLP1, sanscript.ITRANS)
            english_translation = translate_sanskrit_to_english(user_translation)
            user_translation_preprocessed = preprocess_text(english_translation)

            df['Preprocessed_English'] = df['English_Translation'].apply(preprocess_text)

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['Preprocessed_English'])

            user_tfidf = vectorizer.transform([user_translation_preprocessed])

            similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

            max_similarity_index = similarities.argmax()

            if similarities[max_similarity_index] >= 0.3:
                row = df.iloc[max_similarity_index]

                result = {
                    'Aerodynamics': row['Aerodynamics_Concepts'],
                    'Flight_Dynamics': row['Flight_Dynamics_Concepts'],
                    'Drone_Dynamics': row['Drone_Dynamics_Concepts'],
                    'Degree_of_Freedom': row['Degree_of_Freedom'],
                    'Rocket_dynamics': row['Rocket_dynamics'],
                    'NEP': row['NEP']
                }
            else:
                result = {
                    'Aerodynamics': 'No correlation found',
                    'Flight_Dynamics': 'No correlation found',
                    'Drone_Dynamics': 'No correlation found',
                    'Degree_of_Freedom': 'No correlation found',
                    'Rocket_dynamics': 'No correlation found',
                    'NEP': 'No correlation found'
                }

            st.header("Correlations:")
            st.subheader("Aerodynamics:")
            st.write(result['Aerodynamics'])
            st.subheader("Flight dynamics:")
            st.write(result['Flight_Dynamics'])
            st.subheader("Drone dynamics:")
            st.write(result['Drone_Dynamics'])
            st.subheader("Rocket dynamics:")
            st.write(result['Rocket_dynamics'])
            st.subheader("Degree of Freedom:")
            st.write(result['Degree_of_Freedom'])
            st.subheader("National Education Policy:")
            st.write(result['NEP'])

        process_vimanika_shastra(vimanika_df, sanskrit_text)

if selected_domain == "Astro-Physics":
    st.title("Correlation between Astro-Physics and Ancient Sanskrit Shlokas")
    # User input for Sanskrit shloka
    sanskrit_text = st.text_input("Enter the Sanskrit shloka:")

    if st.button("Find Correlations"):
        def preprocess_text(text):
            if isinstance(text, str):
                text = text.lower()
                text = text.translate(str.maketrans("", "", string.punctuation))
            return text

        def translate_sanskrit_to_english(text):
            translator = Translator()
            
            detected_language = translator.detect(text).lang
            
            if detected_language == 'sa':
                result = translator.translate(text, dest='en')
                return result.text
            else:
                return "Invalid source language. Please provide Sanskrit text."

        def display_attributes(df, user_translation):
            # Preprocess the user input
            user_translation_preprocessed = preprocess_text(user_translation)

            # Preprocess the entire 'Shloka' column in the DataFrame
            df['Preprocessed_Sanskrit'] = df['Shloka'].apply(preprocess_text)

            # Drop rows with NaN values in the 'Preprocessed_Sanskrit' column
            df = df.dropna(subset=['Preprocessed_Sanskrit'])

            if df.empty:
                st.warning("No valid data for processing.")
                return

            # Use TfidfVectorizer to convert text to numerical format
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df['Preprocessed_Sanskrit'])

            # Convert user input to numerical format
            user_tfidf = vectorizer.transform([user_translation_preprocessed])

            # Calculate cosine similarities
            similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

            # Find the index of the maximum similarity
            max_similarity_index = similarities.argmax()

            if similarities[max_similarity_index] >= 0.8:
                row = df.iloc[max_similarity_index]

                result = {
                    'Celestial Objects/Mechanics': row['Celestial Objects/Mechanics'],
                    'Ancient Astronomical Observations': row['Ancient Astronomical Observations'],
                    'Modern Scientific Understanding': row['Modern Scientific Understanding'],
                    'Correlation Concepts': row['Correlation Concepts'],
                    'NEP Correlation': row['NEP Correlation']
                    
                }

                # Display the attributes
                st.header("Attributes based on input shloka:")
                st.subheader("Celestial Objects/Mechanics:")
                st.write(result['Celestial Objects/Mechanics'])
                st.subheader("Ancient Astronomical Observations:")
                st.write(result['Ancient Astronomical Observations'])
                st.subheader("Modern Scientific Understanding:")
                st.write(result['Modern Scientific Understanding'])
                st.subheader("Correlation Concepts:")
                st.write(result['Correlation Concepts'])
                
                # Adding NEP Correlation
                st.subheader("NEP Correlation:")
                st.write(result['NEP Correlation'])
            else:
                st.warning("No matching shloka found.")

        display_attributes(astro_df, sanskrit_text)

elif selected_domain == "Vedic Maths":
    st.title("Vedic Maths Calculator")

    def vertical_multiplication(num1, num2):
        num1_str, num2_str = str(num1), str(num2)

        if '.' in num1_str or '.' in num2_str:
            num1_dec, num2_dec = Decimal(num1_str), Decimal(num2_str)
            result_dec = num1_dec * num2_dec
            return result_dec
        else:
            result = [0] * (len(num1_str) + len(num2_str))

            for i in range(len(num1_str) - 1, -1, -1):
                for j in range(len(num2_str) - 1, -1, -1):
                    crosswise_product = int(num1_str[i]) * int(num2_str[j])
                    result[i + j + 1] += crosswise_product

            for i in range(len(result) - 1, 0, -1):
                if result[i] >= 10:
                    carry = result[i] // 10
                    result[i] %= 10
                    result[i - 1] += carry

            result_str = ''.join(map(str, result))
            return int(result_str)

    def multiply_using_loops(a, b):
        result = 0
        a_str = str(a)
        b_str = str(b)

        a_decimals = len(a_str.split('.')[-1]) if '.' in a_str else 0
        b_decimals = len(b_str.split('.')[-1]) if '.' in b_str else 0

        a = int(a * 10 ** a_decimals)
        b = int(b * 10 ** b_decimals)

        if a > b:
            a, b = b, a

        for i in range(abs(a)):
            result += abs(b)

        result /= 10 ** (a_decimals + b_decimals)

        if (a < 0 and b > 0) or (a > 0 and b < 0):
            result = -result

        return result

    # Get user input
    num1 = st.number_input("Enter the first number (a):", value=2.5)
    num2 = st.number_input("Enter the second number (b):", value=3.5)

    # Execute and compare algorithms
    st.header("Results")

    # Vertical Multiplication
    start_time_vertical = timeit.default_timer()
    result_vertical = vertical_multiplication(num1, num2)
    end_time_vertical = timeit.default_timer()
    st.write(f"Vertical Multiplication Result: {result_vertical}")
    st.write(f"Vertical Multiplication Execution Time: {end_time_vertical - start_time_vertical} seconds")

    # Multiplication Using Loops
    start_time_loops = timeit.default_timer()
    result_loops = multiply_using_loops(num1, num2)
    end_time_loops = timeit.default_timer()
    st.write(f"Multiplication Using Loops Result: {result_loops}")
    st.write(f"Multiplication Using Loops Execution Time: {end_time_loops - start_time_loops} seconds")

    # Compare and display the better approach
    st.header("Comparison")
    if result_vertical < result_loops:
        st.success("Vertical Multiplication is better!")
    elif result_loops < result_vertical:
        st.success("Multiplication Using Loops is better!")
    else:
        st.warning("Both algorithms have similar results.")


