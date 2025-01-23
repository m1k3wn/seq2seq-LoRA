import json
import re
from pathlib import Path

def convert_to_uk_spelling(text):
    """Convert various forms of 'symbolize' to UK spelling."""
    replacements = {
        r'symbolize\b': 'symbolise',
        r'symbolizes\b': 'symbolises',
        r'symbolized\b': 'symbolised',
        r'symbolizing\b': 'symbolising'
    }
    
    # Each key in the replacements dict is a US spelling pattern
    for pattern, uk_spelling in replacements.items():
        # Apply the regex substitution for each pattern
        text = re.sub(pattern, uk_spelling, text)
    
    return text

def process_file(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each dream entry
    for entry in data:
        if 'response' in entry:
            entry['response'] = convert_to_uk_spelling(entry['response'])
    
    # Write the modified data back to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Define paths relative to root
    input_file = root_dir / 'data' / 'dreams_data.json'
    output_file = root_dir / 'data' / 'dreams_data_uk.json'
    
    try:
        process_file(input_file, output_file)
        print(f"Successfully converted spellings and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()