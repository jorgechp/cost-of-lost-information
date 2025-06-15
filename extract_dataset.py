"""
Dataset Extraction Tool
=====================

This module extracts and processes academic paper data from JSONL files,
creating a filtered subset with metadata, sections, references, and URIs.

Features:
---------
- JSONL file processing
- Reference extraction
- URI normalization
- Section organization
- Metadata filtering
- Year-based organization
- Optional text content inclusion

Required Dependencies:
--------------------
- json
- argparse
- os
- re
- shutil
- urllib
"""

import argparse
import json
import os
import re
import shutil
from urllib.parse import urlparse
from typing import Dict, List, Optional


class URIProcessor:
    """Handles URI processing and normalization."""

    RSTRIP_URL_CHARACTERS = '),].'

    @staticmethod
    def fix_uri(uri: str) -> Optional[str]:
        """
        Normalize and fix malformed URIs.

        Args:
            uri: URI string to process

        Returns:
            Normalized URI or None if invalid
        """
        try:
            # Fix doubled URLs
            match = re.match(r'([a-zA-Z]+://[^\]]+)\](\1)', uri)
            parsed_uri = match.group(1) if match else uri

            # Clean and parse
            cleaned_uri = parsed_uri.rstrip(URIProcessor.RSTRIP_URL_CHARACTERS)
            parsed = urlparse(cleaned_uri)

            # Rebuild URL
            return parsed._replace(
                path=parsed.path.rstrip(URIProcessor.RSTRIP_URL_CHARACTERS)
            ).geturl()

        except (ValueError, AttributeError):
            print(f"Warning: Invalid URI format: {uri}")
            return None


class PaperProcessor:
    """Processes academic paper data."""

    def __init__(self, include_text: bool = False, arxiv_only: bool = False):
        """
        Initialize processor.

        Args:
            include_text: Whether to include paragraph text
            arxiv_only: Whether to filter for arXiv references only
        """
        self.include_text = include_text
        self.arxiv_only = arxiv_only
        self.uri_processor = URIProcessor()

    def extract_metadata(self, paper: Dict) -> Dict:
        """
        Extract paper metadata.

        Args:
            paper: Raw paper data

        Returns:
            Processed metadata
        """
        metadata = paper.get("metadata", {})
        update_date = metadata.get("update_date", "")
        year = update_date.split('-')[0] if update_date else "unknown"

        return {
            "paper_id": paper.get("paper_id"),
            "title": metadata.get("title"),
            "authors": metadata.get("authors_parsed"),
            "year": year,
            "doi": metadata.get("doi")
        }

    def process_references(self,
                           cite_spans: List[Dict],
                           bib_entries: Dict) -> List[Dict]:
        """
        Process citation references.

        Args:
            cite_spans: List of citation spans
            bib_entries: Bibliography entries

        Returns:
            Processed references
        """
        references = []

        for cite_span in cite_spans:
            ref_id = cite_span.get("ref_id")
            if not ref_id:
                continue

            entry = bib_entries.get(ref_id, {})
            ids = entry.get("ids", {})

            # Skip if arxiv_only and no arXiv ID
            if self.arxiv_only and not ids.get("arxiv_id"):
                continue

            references.append({
                "arxiv_id": ids.get("arxiv_id"),
                "doi": ids.get("doi"),
                "bib_entry_raw": entry.get("bib_entry_raw")
            })

        return references

    def process_section(self, section: Dict, bib_entries: Dict) -> Dict:
        """
        Process paper section.

        Args:
            section: Raw section data
            bib_entries: Bibliography entries

        Returns:
            Processed section
        """
        # Extract URIs
        text = section.get("text", "")
        uri_pattern = re.compile(r'https?://\S+')
        uris = [
            uri for uri in map(self.uri_processor.fix_uri, uri_pattern.findall(text))
            if uri is not None
        ]

        return {
            "section_number": section.get("sec_number"),
            "section_name": section.get("section"),
            "paragraphs": [{
                "text": text if self.include_text else "",
                "cited_references": self.process_references(
                    section.get("cite_spans", []),
                    bib_entries
                ),
                "external_uris": uris
            }]
        }

    def process_paper(self, paper: Dict) -> Dict:
        """
        Process complete paper.

        Args:
            paper: Raw paper data

        Returns:
            Processed paper data
        """
        metadata = self.extract_metadata(paper)
        sections_dict = {}

        for section in paper.get("body_text", []):
            sec_number = section.get("sec_number")
            if sec_number not in sections_dict:
                sections_dict[sec_number] = self.process_section(
                    section,
                    paper.get("bib_entries", {})
                )
            else:
                sections_dict[sec_number]["paragraphs"].extend(
                    self.process_section(section, paper.get("bib_entries", {}))["paragraphs"]
                )

        metadata["sections"] = list(sections_dict.values())
        return metadata


class DatasetExtractor:
    """Handles dataset extraction and file operations."""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 include_text: bool = False,
                 arxiv_only: bool = False):
        """
        Initialize extractor.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            include_text: Whether to include text content
            arxiv_only: Whether to filter for arXiv references
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processor = PaperProcessor(include_text, arxiv_only)

    def setup_output_directory(self) -> None:
        """Create output directory structure."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def process_file(self,
                     input_path: str,
                     output_path: str) -> None:
        """
        Process single JSONL file.

        Args:
            input_path: Input file path
            output_path: Output file path
        """
        with open(input_path, 'r') as in_f, open(output_path, 'w') as out_f:
            for line in in_f:
                try:
                    paper = json.loads(line)
                    processed = self.processor.process_paper(paper)
                    out_f.write(json.dumps(processed) + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {input_path}")
                except Exception as e:
                    print(f"Error processing paper in {input_path}: {str(e)}")

    def extract(self) -> None:
        """Extract and process complete dataset."""
        self.setup_output_directory()

        print(f"Processing files in {self.input_dir}...")
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue

                year = os.path.basename(root)
                output_year_dir = os.path.join(self.output_dir, year)
                os.makedirs(output_year_dir, exist_ok=True)

                self.process_file(
                    os.path.join(root, file),
                    os.path.join(output_year_dir, file)
                )

        # Copy README if exists
        readme_src = os.path.join('resources', 'subset_readme.md')
        if os.path.exists(readme_src):
            shutil.copyfile(
                readme_src,
                os.path.join(self.output_dir, 'README.md')
            )

        print("Dataset extraction complete.")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract academic paper dataset from JSONL files."
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help="Input directory containing JSONL files"
    )
    parser.add_argument(
        'output_directory',
        type=str,
        help="Output directory for processed files"
    )
    parser.add_argument(
        '--text',
        action='store_true',
        help="Include paragraph text content"
    )
    parser.add_argument(
        '--arxiv_only',
        action='store_true',
        help="Include only arXiv references"
    )

    args = parser.parse_args()

    try:
        extractor = DatasetExtractor(
            args.input_directory,
            args.output_directory,
            args.text,
            args.arxiv_only
        )
        extractor.extract()
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    main()