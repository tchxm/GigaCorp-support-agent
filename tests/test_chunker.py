"""
tests/test_chunker.py — Tests for FAQ parsing and document creation.
"""

import os
import pytest
from rag_core import parse_sections, sections_to_documents

FAQ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gigacorp_faq.txt")


class TestParseSections:
    """Verify parse_sections correctly splits the FAQ file."""

    def test_finds_all_sections(self):
        sections = parse_sections(FAQ_PATH)
        headers = [s.header for s in sections]
        assert "SHIPPING POLICY" in headers
        assert "RETURN AND REFUND PROCESS" in headers
        assert "BUSINESS HOURS" in headers
        assert "SERVICE TIERS" in headers
        assert len(sections) == 4

    def test_line_ranges_are_one_indexed(self):
        sections = parse_sections(FAQ_PATH)
        # First section must start at line 1
        assert sections[0].start_line == 1

    def test_line_ranges_contiguous(self):
        sections = parse_sections(FAQ_PATH)
        for i in range(len(sections) - 1):
            # End of section i should be immediately before start of section i+1
            assert sections[i].end_line == sections[i + 1].start_line - 1, (
                f"Gap between sections '{sections[i].header}' "
                f"(end={sections[i].end_line}) and '{sections[i+1].header}' "
                f"(start={sections[i+1].start_line})"
            )

    def test_line_ranges_non_overlapping(self):
        sections = parse_sections(FAQ_PATH)
        for i in range(len(sections) - 1):
            assert sections[i].end_line < sections[i + 1].start_line

    def test_last_section_ends_at_file_end(self):
        sections = parse_sections(FAQ_PATH)
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            total_lines = len(f.readlines())
        assert sections[-1].end_line == total_lines

    def test_sections_have_content(self):
        sections = parse_sections(FAQ_PATH)
        for sec in sections:
            assert len(sec.content.strip()) > 0, (
                f"Section '{sec.header}' has empty content"
            )

    def test_headers_are_clean(self):
        """Headers should NOT contain the '## ' prefix."""
        sections = parse_sections(FAQ_PATH)
        for sec in sections:
            assert not sec.header.startswith("##")
            assert not sec.header.startswith("# ")


class TestSectionsToDocuments:
    """Verify metadata round-trips onto Documents correctly."""

    def test_document_count_matches_sections(self):
        sections = parse_sections(FAQ_PATH)
        docs = sections_to_documents(sections, "gigacorp_faq.txt")
        assert len(docs) == len(sections)

    def test_metadata_fields_present(self):
        sections = parse_sections(FAQ_PATH)
        docs = sections_to_documents(sections, "gigacorp_faq.txt")
        for doc in docs:
            assert "source" in doc.metadata
            assert "section" in doc.metadata
            assert "start_line" in doc.metadata
            assert "end_line" in doc.metadata

    def test_metadata_values_match(self):
        sections = parse_sections(FAQ_PATH)
        docs = sections_to_documents(sections, "gigacorp_faq.txt")
        for sec, doc in zip(sections, docs):
            assert doc.metadata["source"] == "gigacorp_faq.txt"
            assert doc.metadata["section"] == sec.header
            assert doc.metadata["start_line"] == sec.start_line
            assert doc.metadata["end_line"] == sec.end_line

    def test_page_content_includes_header(self):
        sections = parse_sections(FAQ_PATH)
        docs = sections_to_documents(sections, "gigacorp_faq.txt")
        for sec, doc in zip(sections, docs):
            assert sec.header in doc.page_content
