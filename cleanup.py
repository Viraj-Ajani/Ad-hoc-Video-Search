# Input and output file names
INPUT_FILE = 'results.txt'
OUTPUT_FILE = 'cleaned_results.txt'

def clean_shot_id(shot_str):
    # Find the part starting with 'shot'
    start_idx = shot_str.find('shot')
    if start_idx == -1:
        return shot_str  # Return as is if 'shot' is not found

    # Get substring starting from 'shot'
    shot_part = shot_str[start_idx:]

    # Keep only the first two parts separated by underscores (e.g., shot06874_115)
    parts = shot_part.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else shot_part

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # Skip malformed lines
            query_id, raw_shot = parts
            cleaned_shot = clean_shot_id(raw_shot)
            outfile.write(f"{query_id} {cleaned_shot}\n")

if __name__ == "__main__":
    clean_file(INPUT_FILE, OUTPUT_FILE)
    print(f"Cleaned file written to {OUTPUT_FILE}")

