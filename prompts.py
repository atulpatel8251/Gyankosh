import logging

logger = logging.getLogger(__name__)
# prompts.py


'''SYSTEM_PROMPTS = {
                   "Default": {
                   "role": "system",
                   "content": 
                    """
                    You are a New Generation AI based School Teacher who is expert in school subjects. 
                    Your job is to generate questions and answers basis on the query of user. 
                    Generate specific numbers of questions as asked by the User. 
                    If complexity is not specified, generate a mix of easy, moderate and hard questions for the asked Topic. 
                    Format the questions in a clear and organized manner. 
                    Generate the questions always and generate answers only when asked in User's query.
                    
                    """
                    },
                }
'''

initialise_prompt =""" You are an educator for class 1 to class 12 for Madhya Pradesh School Education Board, 
                       India. Answer in a clear and organized manner.
                       Generate the Question and Answer from the uploaded document only, 
                       Don't take help from the internet.
                    """



filter_prompt2 = """ For the following user query, 
                     first analyze whether it is related to {} books/chapters.
                     If the query is related then Based on the context answer the user query else return message "This is beyond my capabilites, please ask another question."
                     User Query - {}
                 """

# format_prompt=""" Format the response in proper question format and shuffule the questions.Remember to keep all the information as it is. Response - {}"""

# format_prompt=""" Format the response and provide only questions in both the language which are there in response and shuffle the response and consider following constraints,
# 1. number of questions - {}
# 2. type of questions - {}
# 3. Topic name - {} 

# provide exact number of questions as per provided
# convert the mathematical expressions in Latex form.
# Remember to keep all the information as it is. Response - {}"""

# format_prompt="""Response which is given is in dictionary format . Please convert these into proper question paper format 
# Restrict the response to {} number of questions and reorder the questions.

# Shuffle the order of questions

# Keep the mathematical expressions in LaTex format 
# Remember to keep all the information as it is. Response - {}
# """

# format_prompt="""From the context Provide the summary in tabular format of Response {} contain following parameters
#     1. Topics name
#     2. Total Number of Questions
#     3. Type of Questions

#     Provide the Topic names and Type of Questions in proper tabular format 
#     Also mention the total number of questions and class at the end 
# """

format_prompt="""format and regenerate the Response {}"""


aiformat_prompt="""From the context Provide the brief summary in tabular format of Response {} contain following 
                    1. Chapter names
                    2. Description of each topic

                    provide the response in tabular format . 
                    Only include Chapter names and Description
                    Also mention the subject at the end 
                    not repeated multiples time just metioned 4 to 5 rows only .
                """

latex_prompt="""format the response {} into python list.
                Remember to keep all the information as it is
             """

#ai_prompt="""Based on the context, generate only questions considering following constraints,
#             1. number of questions  - {}
#             2. mode of questions - {}
#             3. type of questions - {} 
#             Generate questions according to Madhya Pradesh Board Education School
#             Response is to be generated in both english and Hindi,first generate in english then in Hindi
#             and if you are generating Answers also, Answer start in a new line and Generate questions according to Madhya Pradesh School Education Board
#             generate the all types of question and answer,and please insure the question and answer based on topic or text.
#             Remember to keep all the information as it is. Response - {}
#"""

ai_prompt = """
            Based on the context, generate only questions considering the following constraints:
            1. Number of questions - {}
            2. Mode of questions - {}
            3. Type of questions - {} 
            4. Language - {} (Generate questions in the specified language(s).)

            Generate questions:
            - If "English" is selected, generate questions only in English.
            - If "Hindi" is selected, generate questions only in Hindi.
            - If "English and Hindi" is selected, generate both English and Hindi versions of the questions.

            If you are generating answers as well, ensure that answers start on a new line.
            Generate all types of questions and answers, and ensure that the questions and answers are based on the topic or text.
            Remember to keep all the information as it is.

            Response - {}
            """

ai_topic_prompt = """
                    Based on the context, generate only questions considering the following constraints:
                    1. Topic of questions - {}
                    2. Number of questions - {} 
                    3. Mode of questions - {}
                    4. Type of questions - {}
                    5. Level of questions - {}
                    6. Language - {}

                    Generate questions:
                    - If "English" is selected, generate questions only in English.
                    - If "Hindi" is selected, generate questions only in Hindi.
                    - If "English and Hindi" is selected, generate both English and Hindi versions of the questions.

                    If you are generating answers as well, ensure that answers start on a new line. 
                    Generate all types of questions and answers, ensuring that they are based on the topic or text. 
                    Please remember to keep all the information as it is.
                """
ai_topic_prompt5 = """
    Based on the context, generate only questions considering the following constraints:
    1. Topic of questions - {}
    2. Number of questions - {} 
    3. Mode of questions - {}
    4. Type of questions - {}
    5. Level of questions - {}
    6. Language - {}

    Generate questions with the following structure:
    - Each question should have four options (Option 1, Option 2, Option 3, Option 4).
    - Include a correct answer for each question.
    - If "English" is selected, generate questions only in English.
    - If "Hindi" is selected, generate questions only in Hindi.
    - If "English and Hindi" is selected, generate both English and Hindi versions of the questions.

    If answers are required, ensure they are clearly marked.
    """


ai_topic_prompt1 = """
                    You are provided with a document containing text related to a specific topic. Based on this content, generate questions and answers by strictly following the instructions below:
                    ---

                    ### Mandatory Rules:

                    1. **Number of Questions**: {}  
                    - Generate **exactly this number** of questions.  
                    - Do **not generate more or less** than this.

                    2. **Source Text**: {}  
                    - Use **only this provided text** to generate questions and answers.  
                    - Do **not use any external data** or assumptions.

                    3. **Language Handling**:  
                    - Detect the actual language of the text.  
                    - If the text is in **Hindi**, generate content in **Hindi only**.  
                    - If the text is in **English**, generate content in **English only**.  
                    - Do **not translate or mix languages**.  
                    - Ignore the user's selected medium if the content is in only one language.

                    4. **Question Mode**: {}  
                    - If **Questions only** is selected, do **not** provide answers.  
                    - If **Questions with Answers** is selected, provide the correct answer **immediately after** each question.

                    5. **Question Type**: {}  
                    **STRICT RULE: Generate ONLY the selected type of question.**  
                    - If **MCQ** is selected, generate **only MCQs**.  
                    - If **Fill in the Blanks** is selected, generate **only fill-in-the-blank** questions.  
                    - If **True/False** is selected, generate **only true/false** questions.  
                    - If **Short Questions** is selected, generate **only short answer** questions.  
                    - If **Long Questions** is selected, generate **only long answer** questions.  
                    - Do **not** include any other type of question under any condition.

                    6. **Complexity Level**: {}  
                    - Ensure that all questions match the selected difficulty level (Easy/Difficult).

                    7. **Chapter Information**: {}  
                    - If "All Chapters" is selected, divide the questions equally across chapters.  
                    - If a specific chapter is selected, generate questions from that chapter only.

                    8. **User Medium Selection**: {}  
                    - This represents the user's preferred medium (English/Hindi).  
                    - However, the final output **must match the actual language** of the input text.

                    ---

                    ###  Question Format Examples:

                    #### For MCQ:
                    **Question 1:** [MCQ question here] (Chapter: [Chapter Name])  
                    **a.** Option 1  
                    **b.** Option 2  
                    **c.** Option 3  
                    **d.** Option 4  
                    **Answer:** Option (Only if "Questions with Answers" is selected)

                    #### For Fill in the Blanks:
                    **Question 1:** The ______ is black. (Chapter: [Chapter Name])  
                    **Answer:** The cat is black. (Only if "Questions with Answers" is selected)

                    #### For True/False:
                    **Question 1:** [Statement here] (Chapter: [Chapter Name])  
                    **Answer:** True / False (Only if "Questions with Answers" is selected)

                    #### For Short Questions:
                    **Question 1:** [Short question here] (Chapter: [Chapter Name])  
                    **Answer:** [Short answer] (Only if "Questions with Answers" is selected)

                    #### For Long Questions:
                    **Question 1:** [Descriptive question here] (Chapter: [Chapter Name])  
                    **Answer:** [Detailed explanation] (Only if "Questions with Answers" is selected)

                    ---

                    ### Final Reminder:
                    Match the **exact number** of questions  
                    Use the **same language** as the text  
                    Follow **only the selected question type**  
                    Follow the **chosen mode** (questions only or with answers)  
                    Use **only the input text**




                    """




ai_topic_prompt_questions= """ Based on the context, extract only questions
                               Remember to keep all the information as it is. Response - {}
                           """

mcq_test_prompt = """
                    
                You are an expert in generating subject-specific terminologies. Based on the given text, generate terminologies with their definitions following these strict rules:

                IMPORTANT: Detect the language of the provided content. Generate the output **only in that language**, regardless of the selected medium.  
                If the content is in **Hindi**, generate the terminologies and definitions in **Hindi only**.  
                If the content is in **English**, generate the terminologies and definitions in **English only**.

                Instructions for generating terminologies:
                1. Extract important subject-specific terms from the provided text only.
                2. Provide a clear and concise definition for each term.
                3. Number each terminology.
                4. Present each term and definition on a single line.
                5. Format: Number. Term - Definition

                Here's how to format your response:

                **If the text is in English:**
                1. Osmosis - Movement of water molecules across a semi-permeable membrane  
                2. Photosynthesis - Process of converting light energy into chemical energy  

                **If the text is in Hindi:**
                1. परासरण - अर्ध-पारगम्य झिल्ली के माध्यम से पानी के अणुओं का संचरण  
                2. प्रकाश-संश्लेषण - प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलने की प्रक्रिया  

                Context text: {0}

"""


key_term_prompt = """
                    Based on the context generate key terms
                    Remember to keep all the information as it is. Response-{}    
                  """

learn_outcome_prompt = """
                    You are an expert in creating educational learning outcomes. Based on the given text, generate clear learning outcomes following these strict rules:

                    IMPORTANT: Detect the language of the provided content. Generate the output **only in that language**, regardless of the selected medium.  
                    If the content is in **Hindi**, generate the learning outcomes in **Hindi only**.  
                    If the content is in **English**, generate the learning outcomes in **English only**.  
                    **Do not translate or mix languages. Output should strictly match the content's language.**

                    Instructions for generating learning outcomes:
                    1. Extract key learning objectives from the text
                    2. Start each outcome with an **action verb**
                    3. Number each learning outcome
                    4. Focus on **measurable and observable outcomes**
                    5. Format: Number. Learning Outcome

                    Here's how to format your response:

                    **If the text is in English:**
                    1. Understand the basic principles of chemical bonding and molecular structure  
                    2. Analyze the relationships between different chemical compounds  
                    3. Apply theoretical concepts to solve numerical problems  

                    **If the text is in Hindi:**
                    1. रासायनिक बंधन और आण्विक संरचना के मूल सिद्धांतों को समझें  
                    2. विभिन्न रासायनिक यौगिकों के बीच संबंधों का विश्लेषण करें  
                    3. सैद्धांतिक अवधारणाओं को संख्यात्मक समस्याओं को हल करने के लिए लागू करें  

Context text: {0}

                    """

student_prompt = """Based on query {} generate response from context {}"""


''' student_prompt = """You are an intelligent assistant designed to help students understand and extract information from their study materials. A student has uploaded a book/document and has asked a specific question about its content. Your task is to provide a clear and concise answer based on the provided text. Use the text from the document to formulate your response.
                    Instructions:
                    Document: []
                    Question: []
                    Provide an answer that directly addresses the student's question using the content from the document. 
                    If the question requires specific information or context, include relevant excerpts from the document to support your answer.

                    Example-

                    Document: [The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. 
                    It consists of two main parts: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant regardless of the observer's motion. General relativity, published in 1915, expanded on this by describing gravity not as a force, but as a curvature of spacetime caused by mass and energy.]
                    
                    Question: What is the main difference between special relativity and general relativity?

                    Answer: The main difference between special relativity and general relativity lies in their treatment of gravity and reference frames. Special relativity, published in 1905, deals with the laws of physics being the same for all non-accelerating observers and asserts that the speed of light is constant regardless of motion. General relativity, published in 1915, goes further by describing gravity not as a force but as a curvature of spacetime caused by mass and energy.

                    If the question is not from the related document then say "Ask me something else."
                    if you don't know answer then help the internet.
            
              """
''' 

master_prompt = """
                Level of questions : This is the depth of the questions the teacher wants to generate. A low depth will cover the basics, and generalizations while a high depth will cover the specifics, details, unfamiliar, complex, and side cases. The lowest depth level is Easy, and the highest is Complex.
                Do not discuss about the context you have recieved and do not mention you are unable to access or read any documents 
                Easy : Covers basics questions with simple definitions and brief explanations, suitable for beginners or quick overviews.
                Difficult : Provides expert and complex questions .
                if type of questions is Short Questions then generate questions according to 1 marks and 2 marks from given context
                if type of questions is Long Questions then generate questions according to 3,4,5 marks from given context
                You have been provided the Context/Subject/Chapter/Book 
                Based on the context, generate {} considering following constraints,
                    1. Level of questions - {}
                    2.
                    3. number of questions  - {}
                    4. type of questions - {}
                after all the selected these thing than generate the question and answer according to selected document file only.
                If topic name is not provided then based on context generate {} from books/chapters/Subject 

                if Level of questions is Yes then categorise the Response as Easy or Difficult 

                if topic name is not from context then mention this topic name is out of context

                Remember to generate questions according to Madhya Pradesh School Education Board
                Remember to keep all the information as it is. Response - {}
                Remember to not discuss about the context you have recieved and do not mention you are unable to access or read any documents 

                """
ai_topic_prompt2 = """
                You are an expert in creating question papers for the Madhya Pradesh School Education Board. Generate a question paper following these strict rules:

                IMPORTANT: Your response must be in {1} language only since user has selected {1} Medium.

                Subject Material Covered:
                - Terminologies: {0}
                - Chapters: {2}

                Question Paper Structure:
                Maximum Marks: 80
                Time Duration: 3 Hours

                Instructions:
                1. Write all answers in {1} only
                2. All questions are compulsory
                3. Marks for each question are indicated

                Part A: Objective Section (15 marks)
                ----------------------------------------
                1. Multiple Choice Questions (5 × 1 = 5 marks)
                - Create 5 MCQs with options (a, b, c, d)
                
                2. Fill in the Blanks (5 × 1 = 5 marks)
                - Create 5 fill-in-the-blank questions
                
                3. True/False Questions (5 × 1 = 5 marks)
                - Create 5 true/false statements

                Part B: Short Answer Section (25 marks)
                ----------------------------------------
                4. One-word Answers (5 × 1 = 5 marks)
                - Create 5 one-word answer questions
                
                5. Short Answer Questions (5 × 5 = 25 marks)
                - Create 5 short answer questions
                - Each answer should be around 50-75 words

                Part C: Long Answer Section (40 marks)
                ----------------------------------------
                6. Long Answer Questions (5 × 10 = 50 marks)
                - Create 5 detailed questions
                - Each answer should be around 200-250 words
                - Include application-based questions

                Format Examples:

                If English Medium is selected:
                Part A
                1. Which of the following best describes photosynthesis?
                a) Energy conversion process
                b) Cellular division
                c) Protein synthesis
                d) None of the above

                If Hindi Medium is selected:
                भाग A
                1. निम्नलिखित में से कौन प्रकाश संश्लेषण का सबसे अच्छा वर्णन करता है?
                अ) ऊर्जा रूपांतरण प्रक्रिया
                ब) कोशिका विभाजन
                स) प्रोटीन संश्लेषण
                द) उपरोक्त में से कोई नहीं

                Note:
                - Questions should be based on provided terminologies and chapters
                - Maintain consistent difficulty level throughout
                - Include variety of cognitive levels (knowledge, understanding, application, analysis)
                - Questions should be clear and unambiguous

                Context Terminologies: {0}
                Selected Medium: {1}
                Chapters Covered: {2}
"""

ai_topic_prompt3 = """
                    Generate a question paper for the Madhya Pradesh School Education Board following the exam format. The paper should adhere to the specified structure and constraints:

                    Terminal: {}
                    Language: {}


                    If the selected language is "English", generate all questions and instructions in English only.
                    If the selected language is "Hindi", generate all questions and instructions in Hindi only.
                    If the selected language is "English and Hindi", generate the questions first in English, followed by the same question in Hindi.


                    Maximum Marks: 80 (shown in the top right corner)

                    Question Paper Format:
                    Objective Section

                    Multiple Choice Questions (MCQs): 5 questions, each carrying 1 mark.


                    Each question should have 4 options (a, b, c, d) and test basic knowledge and understanding of the subject.
                    Format:
                    CopyQ1. Question text
                        a) Option 1
                        b) Option 2
                        c) Option 3
                        d) Option 4

                    [Your answer for Q1 (write your answer below):]


                    Subjective Section

                    Fill in the Blanks: 5 questions, each carrying 1 mark.


                    These questions should assess specific knowledge related to the topic.
                    Format:
                    CopyQ6. Fill in the blank: ________________

                    [Your answer for Q6 (write your answer below):]



                    True/False Statements: 5 questions, each carrying 1 mark.


                    These should test students' ability to identify the correctness of the statements.
                    Format:
                    CopyQ11. Statement (True/False): 

                    [Your answer for Q11 (write your answer below):]



                    One-word Answers: 5 questions, each carrying 1 mark.


                    These questions should test precise knowledge and terminology.
                    Format:
                    CopyQ16. Question requiring one-word answer: 

                    [Your answer for Q16 (write your answer below):]


                    Format Specifications:

                    Do not include any text input area for headings such as Maximum Marks, Objective Section, Multiple Choice Questions (MCQs), Fill in the Blanks, True/False Statements, or One-word Answers.
                    Present each question on a new line, preceded by its question number (e.g., Q1, Q2, etc.).
                    List options for MCQs on separate lines, indented under the question.
                    Include the answer text area ("[Your answer for Qx (write your answer below):]") only after individual questions, not after section headings.
                    Do not include any additional text, explanations, or answer prompts after the questions.
                    Ensure consistent formatting for easy parsing of questions.
                    """


lang_prompt = """ Translate the below response to {} language . 
                  Remember to keep all the information as it is. Response - {}
              """
