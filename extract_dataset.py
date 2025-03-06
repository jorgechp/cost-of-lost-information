"""
extract_dataset.py

This script extracts a subset of papers from JSONL files, processes the data, and writes the output to a specified directory.
The subset includes metadata, sections, paragraphs, cited references, and external URIs.

Usage:
    python extract_dataset.py <input_directory> <output_directory> [--include_text]

Arguments:
    input_directory (str): The directory containing the input JSONL files.
    output_directory (str): The directory where the output JSONL files will be saved.
    --text (optional): Include the text content of the paragraphs. If not provided, the 'text' field in the paragraph objects will be empty.
    --arxiv_only (optional): Only include cites to articles that exists on arXiv.

Example:
    python extract_dataset.py data/input data/output --include_text

Functions:
    extract_subset(input_dir, output_dir, include_text)
        Extracts a subset of papers from the input directory, filters and processes the data,
        and writes the output to the specified output directory.

    main()
        Parses command-line arguments and calls the extract_subset function.

Dependencies:
    - argparse
    - json
    - os
    - re
    - shutil
"""

import argparse
import json
import os
import re
import shutil
from urllib.parse import urlparse

RSTRIP_URL_CHARACTERS = '),].'


def fix_uri(uri):
    try:
        match = re.match(r'([a-zA-Z]+://[^\]]+)\](\1)', uri)
        parsed_uri = match.group(1) if match else uri
        parsed_uri = urlparse(parsed_uri.rstrip(RSTRIP_URL_CHARACTERS))
        fixed_uri = parsed_uri._replace(path=parsed_uri.path.rstrip(RSTRIP_URL_CHARACTERS)).geturl()
        return fixed_uri
    except ValueError as e:
        print("Error in fix_uri: ", uri)


def extract_subset(input_dir, output_dir, include_text, arxiv_only):
    """
    Extracts a subset of papers from the input directory, filters and processes the data,
    and writes the output to the specified output directory.

    Parameters:
    input_dir (str): The directory containing the input JSONL files.
    output_dir (str): The directory where the output JSONL files will be saved.
    include_text (bool): Whether to include the text content of the paragraphs.
    arxiv_only (bool): Whether to include only references with arXiv IDs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory" + output_dir)

    print(f"Processing files in {input_dir}...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jsonl'):
                year = root.split('/')[-1]
                output_year_dir = os.path.join(output_dir, year)
                if not os.path.exists(output_year_dir):
                    os.makedirs(output_year_dir)

                output_file_path = os.path.join(output_year_dir, file)

                with open(os.path.join(root, file), 'r') as in_f, open(output_file_path, 'w') as out_f:
                    for line in in_f:
                        paper = json.loads(line)
                        update_date = paper.get("metadata", {}).get("update_date", "")
                        year = update_date.split('-')[0] if update_date else "unknown"
                        sections_dict = {}
                        for section in paper.get("body_text", []):
                            sec_number = section.get("sec_number")
                            sec_name = section.get("section")
                            if sec_number not in sections_dict:
                                sections_dict[sec_number] = {
                                    "section_number": sec_number,
                                    "section_name": sec_name,
                                    "paragraphs": []
                                }
                            uri_pattern = re.compile(r'https?://\S+')
                            uri_matches = [fix_uri(uri) for uri in uri_pattern.findall(section.get("text"))]
                            paragraph_info = {
                                "text": section.get("text") if include_text else "",
                                "cited_references": [
                                    {
                                        "arxiv_id": paper.get("bib_entries", {}).get(cite_span.get("ref_id"), {}).get(
                                            "ids", {}).get("arxiv_id"),
                                        "doi": paper.get("bib_entries", {}).get(cite_span.get("ref_id"), {}).get("ids",
                                                                                                                 {}).get(
                                            "doi"),
                                        "bib_entry_raw": paper.get("bib_entries", {}).get(cite_span.get("ref_id"),
                                                                                          {}).get("bib_entry_raw")
                                    }
                                    for cite_span in section.get("cite_spans", [])
                                    if (not arxiv_only or paper.get("bib_entries", {}).get(cite_span.get("ref_id"),
                                                                                           {}).get("ids", {}).get(
                                        "arxiv_id"))

                                ],
                                "external_uris": uri_matches
                            }
                            sections_dict[sec_number]["paragraphs"].append(paragraph_info)
                        sections = list(sections_dict.values())
                        subset = {
                            "paper_id": paper.get("paper_id"),
                            "title": paper.get("metadata", {}).get("title"),
                            "authors": paper.get("metadata", {}).get("authors"),
                            "year": year,
                            "doi": paper.get("metadata", {}).get("doi"),
                            "sections": sections
                        }
                        out_f.write(json.dumps(subset) + '\n')

    readme_src = os.path.join('resources', 'subset_readme.md')
    readme_dst = os.path.join(output_dir, 'README.md')
    shutil.copyfile(readme_src, readme_dst)
    print("Subset extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract a subset of papers from JSONL files.")
    parser.add_argument('input_directory', type=str, help="The directory containing the unarXive directory.")
    parser.add_argument('output_directory', type=str, help="The directory where the subset will be saved.")
    parser.add_argument('--text', action='store_true', help="Include the text content of the paragraphs.")
    parser.add_argument('--arxiv_only', action='store_true', help="Include only references with arXiv IDs.")

    args = parser.parse_args()

    extract_subset(args.input_directory, args.output_directory, args.text, args.arxiv_only)


if __name__ == "__main__":
    main()
