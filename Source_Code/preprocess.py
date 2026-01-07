import pandas as pd
import json
import re
def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def normalize_latex_constraints(text):
    replacements = {
        r'\\leq': ' <= ',
        r'\\le': ' <= ',
        r'\\geq': ' >= ',
        r'\\ge': ' >= ',
        r'\\times': ' * ',
        r'\\cdot': ' * '
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

def clean_text(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    text = normalize_latex_constraints(text)
    text = re.sub(r'\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def preprocess(df):
    text_cols = ['title', 'description', 'input_description', 'output_description']
    for col in text_cols:
        df[col] = df[col].fillna("")

    df['combined_text'] = (
        df['title'] + " " +
        df['description'] + " " +
        df['input_description'] + " " +
        df['output_description']
    )


    df['combined_text'] = df['combined_text'].apply(clean_text)


    return df


if __name__ == "__main__":
    df = load_data("data/problems_data.jsonl")
    df = preprocess(df)
    df.to_csv("data/processed.csv", index=False)
    print("Preprocessing complete ")
