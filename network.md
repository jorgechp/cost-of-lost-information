# Citation Network Format

## Overview

This document describes the format of the citation network exported to a pickle file. The network is represented as a directed graph using the NetworkX library.

## Node Attributes

Each node in the graph represents a paper and contains the following attributes:

- `title`: The title of the paper.
- `year`: The year the paper was published.
- `sections`: A list of sections in the paper. Each section contains:
  - `section_name`: The name of the section.
  - `external_uris`: A list of external URIs found in the section.
  - `cited_references`: A list of cited references in the section.

## Edge Attributes

Each edge in the graph represents a citation from one paper to another. The edge does not contain additional attributes.

## Example Node

Here is an example of a node in the graph:

```json
{
  "title": "Example Paper Title",
  "year": "2021",
  "sections": [
    {
      "section_name": "Introduction",
      "external_uris": ["https://example.com"],
      "cited_references": ["1234.5678"]
    },
    {
      "section_name": "Related Work",
      "external_uris": ["https://relatedwork.com"],
      "cited_references": []
    }
  ]
}