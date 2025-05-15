import os
import pandas as pd
import re

def create_code_df(codes_folder):
    # Directory containing the .txt files with coded sentences

    # Initialize list to collect rows
    rows = []

    codenames = sorted([f for f in os.listdir(codes_folder) if f.endswith(".txt")])

    # Create a mapping from filename to index
    codename_to_id = {fname: idx for idx, fname in enumerate(codenames)}

    # Loop through each file in the folder
    for codename in os.listdir(codes_folder):
        if codename.endswith(".txt"):
            file_path = os.path.join(codes_folder, codename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split content into lines
            lines = content.split('\n')

            # Filter lines that contain at least one alphabetic character and do not begin with "Reference" or "<File"
            filtered_lines = []
            filename = ""
            for line in lines:
                if line.startswith("<File"):
                    # Extract file name
                    filename = re.search(re.search("(?<=\\\\).*(?=>)", line)).group()
                elif re.search(r'[a-zA-Z]', line) and not line.startswith("Reference"):
                    filtered_lines.append((line, filename))

            # Append each filtered line with the corresponding file name
            for line, filename in filtered_lines:
                rows.append([line.strip(), codename_to_id[codename], codename, filename])

    # Create DataFrame
    df_codes = pd.DataFrame(rows, columns=["sentence", "code_no", "code_name", "filename"])
    return df_codes

def create_transcript_df(transcripts_folder):
    # Directory containing the .txt files with transcripts

    # Initialize list to collect rows
    rows = []

    # Loop through each file in the folder
    for transcript in os.listdir(transcripts_folder):
        if transcript.endswith(".txt"):
            file_path = os.path.join(transcripts_folder, transcript)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split content into lines
            responses = content.split('\n')

            # Filter lines that contain at least one alphabetic character and do not begin with "Interviewer"
            filtered_responses = [line for line in responses if re.search(r'[a-zA-Z]', line) and not line.startswith("Interviewer")]

            # Append each filtered line with the corresponding file name
            for response in responses:
                lines = [line.strip() for line in re.split('[,.?!-—:]', response)] # Split by punctuation
                for line in lines:
                    rows.append([line.strip(), os.path.splitext(transcript)[0]])

    # Create DataFrame
    df_transcripts = pd.DataFrame(rows, columns=["line", "filename"])
    return df_transcripts
'''
Given pd dataframes of codes and transcripts, create a dataframe
of each line, the file it appears in, and the codes it's assigned

df_codes has columns=["sentence", "code_no", "code_name", "filename"]
df_transcripts has columns=["line", "file"]

Match based on filename and whether df_transcripts["line"] is a substring of df_codes["sentence"] or vice versa

'''
def create_labeled_transcript_df(df_codes, df_transcripts):
    rows = []
    for index, transcript_row in df_transcripts.iterrows():
        transcript_line, filename = transcript_row["line"], transcript_row["filename"]
        matching_codes_no = []
        matching_codes = []
        # Look for all matching codes ind df_codes with matching filename
        for index, code_row in df_codes["filename" == filename].iterrows():
            code_line, code_no, code_name = code_row["sentence"], code_row["code_no"], code_row["code_name"]
            # Substring match
            if transcript_line in code_line or code_line in transcript_line:
                # Match, add code to matching_codes
                matching_codes_no.append(code_no)
                matching_codes.append(code_name)
            
        # Convert matching codes to string format
        if len(matching_codes_no) == 0:
            matching_codes_no = ""
            matching_codes = ""
        else:
            matching_codes_no = ", ".join(matching_codes_no)
            matching_codes = ", ".join(matching_codes)
        new_row = [transcript_line, matching_codes_no, matching_codes, filename]
        rows.append(new_row)
    # Create DataFrame
    df_transcripts = pd.DataFrame(rows, columns=["text", "label", "codes", "filename"])
    return df_transcripts

'''
Remove all rows in df where the file does not have any codes
'''
def remove_transcripts_with_no_codes(df):
    filtered_df = df.groupby('filename').filter(lambda g: g['label'].str.strip().astype(bool).any())
    return filtered_df

def main():
    codes_folder = input("Insert path name of NVivo codes folder:")
    output_folder = "preprocessed_output"
    df_codes = create_code_df(codes_folder)
    df_codes.to_csv(os.path.join(output_folder, "codes.csv"))

    transcripts_folder = input("Insert path name of transcripts folder:")
    df_transcripts = create_transcript_df(transcripts_folder)
    df_transcripts.to_csv(os.path.join(output_folder, "transcripts_unlabeled.csv"))


    df_labeled_transcripts = create_labeled_transcript_df(df_codes, df_transcripts)
    

    remove_rows = input("Would you like to delete all files with no codes from the dataframe? (y/n)")
    if remove_rows == "y":
        df_labeled_transcripts = remove_transcripts_with_no_codes(df_labeled_transcripts)

    df_labeled_transcripts.to_csv(os.path.join(output_folder, "transcripts_labeled.csv"))

if __name__ == "__main__":
    main()
