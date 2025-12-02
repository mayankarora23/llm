import csv
import random
import string
import json
import pandas as pd

from transformers import AutoTokenizer

def capitalize_where_needed(inputText: str) -> str:
    """
    Capitalizes the first letter of each sentence in the input text.
    This is useful for ensuring that sentences start with a capital letter.
    """
    sentences = inputText.split('. ')
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    result = '. '.join(capitalized_sentences)
    # Only add a period if the final text doesn't already end with one
    #if not result.endswith('.'):
    #    result += '.'
    return result

def capitalize_first_letter(inputText: str) -> str:
    """
    Capitalizes the first letter of the input text.
    This is useful
    for ensuring that the first letter of a sentence is capitalized.
    """
    if not inputText:
        return inputText
    return inputText[0].upper() + inputText[1:]

def captilalize_I_if_needed(inputText: str) -> str:
    """
    Capitalizes the letter 'i' if it appears as a standalone word in the input text.
    This is useful for ensuring that the pronoun 'I' is always capitalized.
    """
    words = inputText.split()
    capitalized_words = [word if word.lower() != 'i' else 'I' for word in words]
    return ' '.join(capitalized_words)

def apply_mask(inputText: str, mask: str = "<extra_id_0>", mask_ratio: float = 0.3) -> str:
    """
    BART-style masking for T5 tokenizer:
    - Each masked span replaced by exactly one <extra_id_0>
    - No consecutive masks
    - Do not mask trailing punctuation
    - Mask ratio ~ mask_ratio
    """
    # Split words while keeping punctuation separate
    import re
    tokens = re.findall(r'\w+|[^\w\s]', inputText, re.UNICODE)
    n = len(tokens)
    masked_tokens = []

    i = 0
    masked_count = 0
    mask_block = 0  # tokens after a mask where masking is forbidden

    while i < n:
        remaining = n - i
        token = tokens[i]

        if mask_block > 0:
            masked_tokens.append(token)
            i += 1
            mask_block -= 1
            continue

        # Only mask alphabetic tokens (skip pure punctuation)
        if token.isalpha() and random.random() < mask_ratio and masked_count < mask_ratio * n:
            span_len = random.randint(3, 6)
            span_len = min(span_len, remaining)

            # Count how many alphabetic tokens in span
            alpha_tokens = [t for t in tokens[i:i+span_len] if t.isalpha()]
            if len(alpha_tokens) == 0:
                masked_tokens.append(token)
                i += 1
                continue

            # Add single mask token
            masked_tokens.append(mask)
            i += span_len
            masked_count += len(alpha_tokens)

            # Avoid consecutive masks
            mask_block = random.randint(1, 2)
        else:
            masked_tokens.append(token)
            i += 1

    # Rejoin tokens with space but avoid space before punctuation
    result = ''
    for t in masked_tokens:
        if t in string.punctuation:
            result += t
        else:
            if result:
                result += ' '
            result += t
    return result

def build_input_target_pair(csvFileName: str, outputFileName: str):
    # Open the CSV file
    with open(csvFileName, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        with open(outputFileName, 'w') as outputFile:
            outputFile.write('{\n  "InputTargetPairs": [\n')

            firstLine = True
            count = 0
            lines = []

            for row in reader:
                if row[0] == "0":
                    continue
                lines.append(row[0])
                count += 1
                if count >= 30000:
                    break

            count = 0

            print(f"Total lines read: {len(lines)}")

            lines.sort(key=lambda x: len(x), reverse=False)  # Sort by length, smallest first

            # Read each row
            for row in lines:
                if row == "0":
                    continue
                line = row.strip()
                line = capitalize_first_letter(line)
                line = captilalize_I_if_needed(line)
                line = capitalize_where_needed(line)
                maskedLine = apply_mask(line, mask_ratio=0.35)

                if not firstLine:
                    outputFile.write(',\n')
                firstLine = False

                outputFile.write(f'    {{\n      "Input": "{maskedLine}",\n      "Target": "{line}<endoftext>"\n    }}')
                count += 1

                if count == 30000:
                    break

            outputFile.write('\n  ]\n}')

def isOnlyPunctuationOrNumber(text: str) -> bool:
    for char in text:
        if char.isalpha():
            return False
    return True

def isTokensEnough(tokenizer, text: str, minTokens: int, maxTokens: int) -> bool:
    tokens = tokenizer.tokenize(text)

    if len(tokens) > minTokens and len(tokens) < maxTokens:
        return True
    return False

def build_input_text(csvFileName: str, outputFileName: str):
    # Open the CSV file
    import sys
    with open(csvFileName, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        csv.field_size_limit(sys.maxsize)

        with open(outputFileName, 'w') as outputFile:
            outputFile.write('{\n  "InputTexts": [\n')
            tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

            firstLine = True
            count = 0
            lines = []

            for row in reader:
                if row[0] == "0" or isOnlyPunctuationOrNumber(row[0]) or not isTokensEnough(tokenizer, row[0], 0, 510):
                    continue
                lines.append(row[0])
                if count % 1000 == 0:
                    print(f"Added line number : {count+1}")
                count += 1
                if count >= 400000:
                    break

            count = 0

            print(f"Total lines read: {len(lines)}")

            #lines.sort(key=lambda x: len(x), reverse=False)  # Sort by length, smallest first

            # Read each row
            for row in lines:
                if row == "0":
                    continue
                line = row.strip()
                line = capitalize_first_letter(line)
                line = captilalize_I_if_needed(line)
                line = capitalize_where_needed(line)
                #print(f"Row : {row}\nLine: {line}")
                maskedLine = apply_mask(line, mask_ratio=0.35)

                if not firstLine:
                    outputFile.write(',\n')
                firstLine = False

                outputFile.write(f'    {{\n      "Index": {count+1},\n      "Text": "{line}"\n    }}')
                count += 1

                #if count == 30000:
                #    break

            outputFile.write('\n  ]\n}')

def check_input_target_pairs(fileName):
    with open(fileName, 'r') as file:
        data = json.load(file)
        pairs = data.get("InputTargetPairs")
        if not pairs:
            print("No input-target pairs found.")
            return

        for pair in pairs:
            input_text = pair.get("Input", "")
            target_text = pair.get("Target", "")
            if not input_text or not target_text:
                print(f"Invalid pair found: {pair}")
            else:
                print(f"Valid pair: Input: {input_text}\n Target: {target_text}")

def check_input_text(fileName):
    with open(fileName, 'r') as file:
        data = json.load(file)
        pairs = data.get("InputTexts")
        if not pairs:
            print("No input-target pairs found.")
            return

        for pair in pairs:
            input_text = pair.get("Text", "")
            if not input_text:
                print(f"Invalid pair found: {pair}")
            else:
                print(f"Valid pair: Input: {input_text}")

def main():
    csvFileName = "BookCorpus3.csv"
    outputFileName = "book_corpus_training_dataset_input_texts.json"

    build_input_text(csvFileName, outputFileName)
    #check_input_text(outputFileName)

if __name__ == "__main__":
    main()