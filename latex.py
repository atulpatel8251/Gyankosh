import streamlit as st
import re

# Define the LaTeX strings

def text_to_latex(question):
    # Convert question to LaTeX format
    latex_question = r"\text{" + question + r"}"
    
    return f"{latex_question}"
    
def add_dollar_signs(text):
    # Regular expression pattern to find equations
    equation_pattern = r"([+-]?\s*\d*\s*[xX]\^\d+)"

    # Find equations in the text using regular expression
    equations = re.findall(equation_pattern, text)

    # Replace each equation with the same equation surrounded by dollar signs
    for equation in equations:
        text = text.replace(equation, f"${equation}$")

    return text

# Define the questions in text format
questions_text = """If we add the polynomials 2x^2 + 4x + 1 and x^2 - 3x + 2, what is the coefficient of x?
                    If we add the polynomials 2x^3 + 5x + 1 and x^2 - 3x + 2, what is the coefficient of x?"""

# Convert questions to LaTeX format
formatted_questions = text_to_latex(questions_text)

# Add dollar signs around equations in the questions
formatted_questions_with_dollars = add_dollar_signs(formatted_questions)

# Display the formatted questions using Streamlit
st.latex(formatted_questions_with_dollars)
