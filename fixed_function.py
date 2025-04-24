# in this code all options must be included in the same paragraph as the question number
import pandas as pd
#from docx import Document
import re
import os
import docx2txt
import pytesseract
from docxlatex import Document 

def is_numbered(text):
    # Check if the text starts with a number followed by a closing parenthesis
    return bool(re.match(r'^\d+\)', text.strip()))

def extract_images_from_docx(docx_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    docx2txt.process(docx_path, output_folder)

    image_filenames = [filename for filename in os.listdir(output_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    #print(image_filenames)

    for i, image_filename in enumerate(image_filenames):
        image_path = os.path.join(output_folder, image_filename)
        text = pytesseract.image_to_string(image_path,config='--psm 4')
        #print("==")
        #print(text)
        fig_name=getname(text)
        #print("figname=",fig_name)
        if os.path.exists(os.path.join(output_folder, fig_name+f".png")):
            pass
        else:  
            new_image_path = os.path.join(output_folder, fig_name+f".png")
            os.rename(image_path, new_image_path)

def image_path(df,path):
    pattern = r'Figure \d+'

    for index, row in df.iterrows():
        question = 'roQuestion'
        fig_match = re.search(pattern, question)
        if fig_match:
            fig_number = fig_match.group()
            # Update the image path with .png extension
            df.at[index, 'Image Path'] = path+"/"+fig_number + '.png'
        else:
            df.at[index, 'Image Path'] = "no_image"
            
    return df

def getname(text):
    fig_name = re.findall(r'Figure \d+', text, re.DOTALL)
    if fig_name:
        return fig_name[0]
    else:
        return '_'
    
def extract_text_by_heading(doc_path):
    doc = Document(doc_path)
    text = doc.get_text()

    lines = text.split('\n')

    data = []

    topic_name = ''
    question_type = ''
    questions_dict = {}

    def is_option_line(line):
        return re.match(r'^\([a-zA-Z]+\)$', line.strip())

    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('Topic:'):
                topic_name = line.replace('Topic:', '').strip()
            elif line.startswith('Type:'):
                question_type = line.replace('Type:', '').strip()
                if topic_name and question_type:
                    questions_dict[(topic_name, question_type)] = []
            elif line[0].isdigit():
                question_parts = line.split(')', 1)
                if len(question_parts) > 1:
                    current_line = question_parts[1].strip()
                else:
                    current_line = line.strip()
                if topic_name and question_type:
                    questions_dict[(topic_name, question_type)].append(current_line)
            elif is_option_line(line) and topic_name and question_type:
                if questions_dict[(topic_name, question_type)]:  # Check if the list is not empty
                    questions_dict[(topic_name, question_type)][-1] += '\n' + line.strip()
            elif topic_name and question_type:
                if questions_dict[(topic_name, question_type)]:  # Check if the list is not empty
                    questions_dict[(topic_name, question_type)][-1] += '\n' + line.strip()

    for (topic, q_type), q_lines in questions_dict.items():
        data.extend([(topic, q_type, q_line) for q_line in q_lines])

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data, columns=['Topic', 'Type of Question', 'Question'])
    return df

def optimize_question_type(question_type):
    return question_type.strip().strip('()')  # Added .strip() to remove spaces

