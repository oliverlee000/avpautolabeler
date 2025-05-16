import os
import pandas as pd
import re
from collections import defaultdict
import argparse

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
            with open(file_path, "r", encoding="Latin-1") as f:
                content = f.read()

            # Split content into lines
            lines = content.split('\n')

            # Filter lines that contain at least one alphabetic character and do not begin with "Reference" or "<File"
            filtered_lines = []
            filename = ""
            for line in lines:
                if line.startswith("<File"):
                    # Extract file name
                    filename = re.search("(?<=\\\\)\w.*(?=>)", line).group()
                elif re.search(r'[a-zA-Z]', line) and not line.startswith("Reference"):
                    if line.startswith("Interviewee:") or line.startswith("Interviewer:"):
                        line = re.search("(?<=Interviewee:|Interviewer:).*", line).group()
                    filtered_lines.append((line, filename))

            # Append each filtered line with the corresponding file name
            for line, filename in filtered_lines:
                rows.append([line, codename_to_id[codename], codename, filename])

    # Create DataFrame
    df_codes = pd.DataFrame(rows, columns=["sentence", "code_no", "code_name", "filename"])
    return df_codes

'''
filter_for_named = TRUE if you just want to add transcripts whose filenames wstarts with a letter

'''
def create_transcript_df(transcripts_folder, filter_for_named=True):
    # Directory containing the .txt files with transcripts

    # Initialize list to collect rows
    rows = []

    # Loop through each file in the folder
    for transcript in os.listdir(transcripts_folder):
        if transcript.endswith(".txt"):
            filename = os.path.splitext(transcript)[0]
            if filter_for_named and not re.match("^[A-Z]", transcript):
                continue
            file_path = os.path.join(transcripts_folder, transcript)
            with open(file_path, "r", encoding="Latin-1") as f:
                content = f.read()

            # Split content into lines
            lines = content.split('\n')
            is_interviewee = False
            for i in range(0, len(lines), 2)
            for line in lines:
                if line.startswith("Interviewer"):
                    is_interviewee = False
                elif line.startswith("Interviewee:"):
                    is_interviewee = True
                    line = re.search("(?<=Interviewee:).*", line).group()
                if is_interviewee:
                    sentences = [s.strip() for s in line.split(".")] # split by period
                    for i in range(0, len(sentences), 2):
                        chunk = ' '.join(sentences[i:i+2]) # Two sentence chunks
                        rows.append([chunk, filename])

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
    code_dict = defaultdict(list)
    for index, code_row in df_codes.iterrows():
        code_line, code_no, code_name, filename = code_row["sentence"], code_row["code_no"], code_row["code_name"], code_row["filename"]
        filtered_df = df_transcripts[df_transcripts["filename"].str.startswith(filename)] # Search through entries with matching filename
        if len(filtered_df) > 0:
            for index, transcript_row in filtered_df.iterrows():
                transcript_line, filename = transcript_row["line"], transcript_row["filename"]
                # Substring match
                print(transcript_line)
                print(code_line)
                if transcript_line in code_line or code_line in transcript_line:
                    # Match, add code to matching_codes
                    code_dict[transcript_line].append((code_no, code_name))
    for key, value in code_dict.items():
        code_no = ""
        code_name = ""
        if len(value) != 0:
            code_no = ", ".join([v[0] for v in value])
            code_name = ", ".join([v[1] for v in value])
        new_row = [transcript_line, code_no, code_name, filename]
        rows.append(new_row)
    '''
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
    '''
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes_df', type=str, help='Path to code dataframe file')
    parser.add_argument('--transcripts_df', type=str, help='Path to unlabeled transcript dataframe file')
    parser.add_argument('--codes_folder', type=str, help='Path to folder that contains NVivo codes as txt files')
    parser.add_argument('--transcripts_folder', type=str, help='Path to folder that contains unlabeled transcripts as txt files')
    parser.add_argument('--output_folder', type=str, help='Output folder to store processed data', default="preprocessed_output")
    parser.add_argument('--remove_empty', action='store_true', help='Remove all rows from transcripts with no codes from df')
    parser.add_argument('--filter_for_named', action='store_true', help='Filter for transcript files whose names begin with an alphabetic character')
    args = parser.parse_args()

    # Create df_codes -> csv file storing all coded text from all transcripts
    if args.codes_df:
        print(f"Using codes_df path: {args.codes_df}")
        df_codes = pd.read_csv(args.codes_df)
    else:
        codes_folder = args.codes_folder if args.codes_folder else input("Insert path name of NVivo codes folder:")
        print(f"Using {codes_folder}")
        df_codes = create_code_df(codes_folder)
        
        # Write CSV
        df_codes_pn = os.path.join(args.output_folder, "codes.csv")
        df_codes.to_csv(df_codes_pn)
        print(f"codes_df created at pathname {df_codes_pn}.")
    
    # Create df_transcripts -> csv file storing all lines from transcripts separated by punctuation mark
    if args.transcripts_df:
        print(f"Using transcripts_df path: {args.transcripts_df}")
        df_transcripts = pd.read_csv(args.transcripts_df)
    else:
        transcripts_folder = args.transcripts_folder if args.transcripts_folder else input("Insert path name of transcripts folder:")
        print(f"Using {transcripts_folder}")
        df_transcripts = create_transcript_df(transcripts_folder, filter_for_named=args.filter_for_named)
        
        # Write CSV
        df_transcripts_pn = os.path.join(args.output_folder, "transcripts_unlabeled.csv")
        df_transcripts.to_csv(df_transcripts_pn)
        print(f"transcripts_unlabeled_df created at pathname {df_transcripts_pn}.")

    # Create df_labeled_transcripts -> csv file storing all lines from transcripts separated by punctuation mark, AND all codes assigned to it
    df_labeled_transcripts = create_labeled_transcript_df(df_codes, df_transcripts)
    
    if args.remove_empty:
        remove_rows = input("Would you like to delete all files with no codes from the dataframe? (y/n)")
        if remove_rows == "y":
            df_labeled_transcripts = remove_transcripts_with_no_codes(df_labeled_transcripts)

    
    # Write CSV
    transcripts_pn = os.path.join(args.output_folder, "transcripts_labeled.csv")
    df_labeled_transcripts.to_csv(transcripts_pn)
    print(f"transcripts_labeled_df created at pathname {transcripts_pn}.")
    print("Done.")

if __name__ == "__main__":
    main()
