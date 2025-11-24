"""O-RAN Specification Parser
Extracts hierarchical structure from O-RAN documents for semantic chunking.
"""

import re
from typing import List, Dict, Tuple


class ORANSectionParser:
    """Parse O-RAN specifications into structured sections."""

    # Enhanced regex patterns for O-RAN specifications
    SECTION_PATTERNS = [
        # Matches: "3.1 C-Plane Messages" or "3.1.2 Beamforming"
        r"^(\d+(?:\.\d+)*)\s+(.+)$",
        # Matches: "ANNEX A: Security" or "Annex B (informative)"
        r"^(ANNEX\s+[A-Z])[\s:]+(.+?)(?:\s*\((?:normative|informative)\))?$",
        # Matches: "SECTION 4 - Protocol Details"
        r"^SECTION\s+(\d+)\s*[-:]+\s*(.+)$",
    ]

    def __init__(self) -> None:
        self.compiled_patterns = [
            re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            for pattern in self.SECTION_PATTERNS
        ]

    def detect_section_header(self, line: str) -> Tuple[bool, str, str]:
        """Detect if a line is a section header."""
        line = line.strip()
        for pattern in self.compiled_patterns:
            match = pattern.match(line)
            if match:
                section_id = match.group(1)
                section_title = match.group(2).strip()
                return True, section_id, section_title
        return False, "", ""

    def parse_document(
        self,
        text: str,
        doc_id: str = "unknown",
        work_group: str = "unknown",
    ) -> List[Dict]:
        """Parse document into structured sections."""
        lines = text.split("\n")
        sections: List[Dict] = []
        current_section = {
            "doc_id": doc_id,
            "work_group": work_group,
            "section_id": "preamble",
            "section_title": "Preamble",
            "content": [],
        }
        for line in lines:
            is_header, section_id, section_title = self.detect_section_header(line)
            if is_header:
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)
                current_section = {
                    "doc_id": doc_id,
                    "work_group": work_group,
                    "section_id": section_id,
                    "section_title": section_title,
                    "content": [],
                }
            else:
                current_section["content"].append(line)
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)
        return sections

    def extract_work_group(self, doc_id: str) -> str:
        """Extract work group from document ID."""
        match = re.search(r"WG(\d+)", doc_id, re.IGNORECASE)
        if match:
            return f"WG{match.group(1)}"
        return "unknown"


def test_parser() -> None:
    """Test the O-RAN parser."""
    sample_text = """
O-RAN ALLIANCE Technical Specification

1 Scope
This document defines the fronthaul interface.

1.1 Overview
The fronthaul connects O-DU and O-RU.

2 References
See 3GPP TS 38.401 for background.

3 Architecture
3.1 C-Plane
The Control Plane uses eCPRI protocol.

3.1.1 Message Format
Messages follow the structure in Table 3.1.

ANNEX A: Security Considerations
This annex is normative.
"""
    parser = ORANSectionParser()
    sections = parser.parse_document(
        sample_text,
        doc_id="O-RAN.WG4.Test",
        work_group="WG4",
    )
    print(f"Parsed {len(sections)} sections:")
    for sec in sections:
        print(
            f"  [{sec['section_id']}] {sec['section_title']} "
            f"({len(sec['content'])} chars)"
        )


if __name__ == "__main__":
    test_parser()
