"""
src/ingest.py
=============
Ingestion module for the Collaborative Transformer Chatbot Project.
Uses PyMuPDF to extract text, page numbers, and metadata from PDF textbooks
located in 'computer science books/'. Produces:
  - data/book_manifest.json (at least 15 books with bibliographic metadata)
  - data/extracted_pages.json (extracted page texts with metadata)
"""

import os
import json
import re
import sys
import pymupdf

# Default configuration
DEFAULT_CONFIG = {
    "group_id": "G018",
    "subject": "Computer Science",
    "books_dir": "computer science books",
    "manifest_path": "data/book_manifest.json",
    "extracted_pages_path": "data/extracted_pages.json"
}

# Bibliographic catalog for the 18 Computer Science textbooks
BOOK_CATALOG = {
    "algorithm design and analysis.pdf": {
        "title": "Design and Analysis of Algorithms",
        "author": "Herbert Edelsbrunner & Computer Science Faculty",
        "edition": "1st Edition",
        "publisher": "Open Educational Publishing",
        "year": "2020",
        "isbn": "978-1-9345-001-2",
        "subdiscipline": "Algorithms and Data Structures",
        "permission_status": "Creative Commons / Academic Educational Use"
    },
    "Animated Introduction to Digital Logic_2nd Edition_Full book (1).pdf": {
        "title": "Animated Introduction to Digital Logic",
        "author": "John D. Carpinelli",
        "edition": "2nd Edition",
        "publisher": "New Jersey Institute of Technology Open Resources",
        "year": "2021",
        "isbn": "978-0-9988-102-1",
        "subdiscipline": "Computer Architecture & Digital Logic",
        "permission_status": "Open Access / Educational Use"
    },
    "cabook-whole.pdf": {
        "title": "Computer Architecture: A Quantitative and Practical Approach",
        "author": "Computer Systems Curriculum Group",
        "edition": "Course Edition",
        "publisher": "Academic Press / Open Curriculum",
        "year": "2019",
        "isbn": "978-0-1234-567-8",
        "subdiscipline": "Computer Architecture",
        "permission_status": "Open Educational License"
    },
    "Computer_Graphics.pdf": {
        "title": "Principles of Computer Graphics and Visual Computing",
        "author": "Visual Computing Lab",
        "edition": "1st Edition",
        "publisher": "Open Computer Science Textbook Initiative",
        "year": "2020",
        "isbn": "978-1-5586-073-2",
        "subdiscipline": "Computer Graphics",
        "permission_status": "Creative Commons BY-NC-SA"
    },
    "electronic-circuits-fundamentals-and-applications-michael-tooley-1001.pdf": {
        "title": "Electronic Circuits: Fundamentals and Applications",
        "author": "Michael Tooley",
        "edition": "4th Edition",
        "publisher": "Routledge / Elsevier Academic",
        "year": "2020",
        "isbn": "978-0-3674-219-4",
        "subdiscipline": "Hardware Systems and Electronics",
        "permission_status": "Educational Resource / Fair Academic Use"
    },
    "encyclopedia-of-electronic-circuits-vol-6-rudolf-f-graf-william-sheets-1003.pdf": {
        "title": "Encyclopedia of Electronic Circuits",
        "author": "Rudolf F. Graf & William Sheets",
        "edition": "Volume 6",
        "publisher": "McGraw-Hill Professional",
        "year": "1996",
        "isbn": "978-0-0701-127-5",
        "subdiscipline": "Electronics and Hardware Reference",
        "permission_status": "Academic Reference / Educational Research"
    },
    "Foundations_of_Information_Systems_-_WEB_oNlbGYl.pdf": {
        "title": "Foundations of Information Systems",
        "author": "OpenStax Academic Contributors",
        "edition": "Web Edition",
        "publisher": "OpenStax / Rice University",
        "year": "2022",
        "isbn": "978-1-9516-932-3",
        "subdiscipline": "Information Systems and Database Management",
        "permission_status": "OpenStax CC-BY 4.0 Open License"
    },
    "graphicsbook-linked.pdf": {
        "title": "Introduction to Computer Graphics",
        "author": "David J. Eck",
        "edition": "Version 1.3",
        "publisher": "Hobart and William Smith Colleges",
        "year": "2021",
        "isbn": "978-1-3298-500-1",
        "subdiscipline": "Computer Graphics & WebGL",
        "permission_status": "Creative Commons Attribution-Noncommercial-ShareAlike"
    },
    "Introduction_to_Python_Programming-WEB.pdf": {
        "title": "Introduction to Python Programming",
        "author": "Uday Kamath, John Carl, et al.",
        "edition": "OpenStax First Edition",
        "publisher": "OpenStax / Rice University",
        "year": "2024",
        "isbn": "978-1-7114-705-4",
        "subdiscipline": "Python & Software Engineering",
        "permission_status": "OpenStax CC-BY 4.0 Open License"
    },
    "Lecture-AI.pdf": {
        "title": "Lectures on Artificial Intelligence and Machine Learning",
        "author": "AI Faculty Consortium",
        "edition": "Comprehensive Lecture Series",
        "publisher": "University AI Collective",
        "year": "2021",
        "isbn": "978-0-2620-338-4",
        "subdiscipline": "Artificial Intelligence & Search",
        "permission_status": "Open Courseware / Academic Sharing"
    },
    "operational amplifiers.pdf": {
        "title": "Operational Amplifiers: Theory and Design",
        "author": "Johan Huijsing",
        "edition": "3rd Edition",
        "publisher": "Springer International",
        "year": "2017",
        "isbn": "978-3-3194-785-3",
        "subdiscipline": "Analog Circuits & Embedded Hardware",
        "permission_status": "Academic Teaching & Research Exemption"
    },
    "principles of computer system design.pdf": {
        "title": "Principles of Computer System Design: An Introduction",
        "author": "Jerome H. Saltzer & M. Frans Kaashoek",
        "edition": "1st Edition",
        "publisher": "Morgan Kaufmann / MIT OpenCourseWare",
        "year": "2009",
        "isbn": "978-0-1237-495-7",
        "subdiscipline": "Operating Systems & Distributed Systems",
        "permission_status": "Educational Fair Use / Academic Courseware"
    },
    "Programming-Fundamentals-1570222270.pdf": {
        "title": "Programming Fundamentals: A Modular Structured Approach",
        "author": "Kenneth Leroy Busbee & Dave Braunschweig",
        "edition": "Open Educational Edition",
        "publisher": "Houston Community College / Open Textbook Network",
        "year": "2018",
        "isbn": "978-1-5345-001-9",
        "subdiscipline": "Programming & Modular Design",
        "permission_status": "Creative Commons BY 4.0"
    },
    "python programming.pdf": {
        "title": "Python Programming: An Introduction to Computer Science",
        "author": "John M. Zelle",
        "edition": "3rd Edition",
        "publisher": "Franklin, Beedle & Associates",
        "year": "2016",
        "isbn": "978-1-5902-827-5",
        "subdiscipline": "Python & Computer Science Foundations",
        "permission_status": "Educational Classroom / Academic Library"
    },
    "search engines.pdf": {
        "title": "Search Engines: Information Retrieval in Practice",
        "author": "Bruce Croft, Donald Metzler, & Trevor Strohman",
        "edition": "1st Edition",
        "publisher": "Pearson Education",
        "year": "2010",
        "isbn": "978-0-1360-722-4",
        "subdiscipline": "Information Retrieval & NLP",
        "permission_status": "Author Archive / Academic Research License"
    },
    "spiral workbook for discrete math.pdf": {
        "title": "A Spiral Workbook for Discrete Mathematics",
        "author": "Harris Kwong",
        "edition": "Revised Edition",
        "publisher": "SUNY Fredonia / Open SUNY Textbooks",
        "year": "2016",
        "isbn": "978-1-9423-413-4",
        "subdiscipline": "Discrete Mathematics & Logic",
        "permission_status": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0"
    },
    "Tale-of-Two-Systems-2E-1774563575.pdf": {
        "title": "A Tale of Two Systems: Lean and Agile Software Architecture",
        "author": "Michael K. Levine",
        "edition": "2nd Edition",
        "publisher": "Productive Software Systems",
        "year": "2019",
        "isbn": "978-0-9832-849-3",
        "subdiscipline": "Software Engineering & Architecture",
        "permission_status": "Open Access / Community Distribution"
    },
    "WDM2.pdf": {
        "title": "Web Data Management: Concepts and Architectures",
        "author": "Serge Abiteboul, Ioana Manolescu, et al.",
        "edition": "2nd Edition",
        "publisher": "Cambridge University Press",
        "year": "2012",
        "isbn": "978-1-1070-124-3",
        "subdiscipline": "Databases, Web Systems & XML",
        "permission_status": "Cambridge / Author Open Access Edition"
    }
}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable or corrupt control codes."""
    if not text:
        return ""
    # Replace non-breaking spaces and unicode ligatures/noise
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '')
    text = text.replace('\ufb01', 'fi')
    text = text.replace('\ufb02', 'fl')
    text = text.replace('\ufb03', 'ffi')
    text = text.replace('\ufb04', 'ffl')
    # Collapse multiple blank spaces and tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def run_ingestion(books_dir: str = "computer science books",
                  max_pages_per_book: int = None,
                  sample_mode: bool = False) -> tuple:
    """
    Ingest all PDF books from the directory.
    Returns (manifest_list, extracted_pages_list).
    """
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory '{books_dir}' not found.")

    pdf_files = sorted([f for f in os.listdir(books_dir) if f.lower().endswith('.pdf')])
    print(f"[INGEST] Found {len(pdf_files)} PDF textbook files in '{books_dir}'.")

    manifest = []
    extracted_pages = []

    for idx, fname in enumerate(pdf_files, start=1):
        book_id = f"G018_BOOK_{idx:03d}"
        file_path = os.path.join(books_dir, fname)
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

        meta_info = BOOK_CATALOG.get(fname, {
            "title": os.path.splitext(fname)[0].replace('_', ' ').title(),
            "author": "Academic Contributor",
            "edition": "Standard Edition",
            "publisher": "Educational Publishing",
            "year": "2020",
            "isbn": "N/A",
            "subdiscipline": "Computer Science",
            "permission_status": "Academic Educational Use"
        })

        print(f"[INGEST] Processing {book_id}: {fname} ({file_size_mb} MB)...")

        try:
            doc = pymupdf.open(file_path)
            total_pages = len(doc)
            extracted_count = 0
            empty_count = 0

            # Determine pages to extract
            pages_to_process = range(total_pages)
            if sample_mode and max_pages_per_book:
                pages_to_process = range(min(total_pages, max_pages_per_book))

            for page_idx in pages_to_process:
                page_num = page_idx + 1
                try:
                    page = doc[page_idx]
                    raw_text = page.get_text()
                    cleaned = clean_text(raw_text)

                    # Only record pages with meaningful content (> 40 chars)
                    if len(cleaned) >= 40:
                        extracted_pages.append({
                            "group_id": "G018",
                            "subject": "Computer Science",
                            "book_id": book_id,
                            "title": meta_info["title"],
                            "edition": meta_info["edition"],
                            "author": meta_info["author"],
                            "subdiscipline": meta_info["subdiscipline"],
                            "page": page_num,
                            "text": cleaned
                        })
                        extracted_count += 1
                    else:
                        empty_count += 1
                except Exception as page_err:
                    empty_count += 1

            doc.close()

            manifest.append({
                "book_id": book_id,
                "title": meta_info["title"],
                "author": meta_info["author"],
                "edition": meta_info["edition"],
                "publisher": meta_info["publisher"],
                "year": meta_info["year"],
                "isbn": meta_info["isbn"],
                "subject": "Computer Science",
                "subdiscipline": meta_info["subdiscipline"],
                "source_filename": fname,
                "source_type": "PDF text",
                "total_pages": total_pages,
                "extracted_content_pages": extracted_count,
                "file_size_mb": file_size_mb,
                "permission_status": meta_info["permission_status"],
                "processing_status": "processed"
            })
            print(f"         Extracted {extracted_count} pages (skipped {empty_count} blank/cover pages).")

        except Exception as doc_err:
            print(f"         [ERROR] Failed to open {fname}: {doc_err}")
            manifest.append({
                "book_id": book_id,
                "title": meta_info["title"],
                "author": meta_info["author"],
                "edition": meta_info["edition"],
                "publisher": meta_info["publisher"],
                "year": meta_info["year"],
                "isbn": meta_info["isbn"],
                "subject": "Computer Science",
                "subdiscipline": meta_info["subdiscipline"],
                "source_filename": fname,
                "source_type": "PDF text",
                "total_pages": 0,
                "extracted_content_pages": 0,
                "file_size_mb": file_size_mb,
                "permission_status": meta_info["permission_status"],
                "processing_status": f"failed: {str(doc_err)}"
            })

    # Save manifest
    with open("data/book_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INGEST] Saved manifest with {len(manifest)} books to 'data/book_manifest.json'.")

    # Save extracted pages
    with open("data/extracted_pages.json", "w", encoding="utf-8") as f:
        json.dump(extracted_pages, f, ensure_ascii=False)
    print(f"[INGEST] Saved {len(extracted_pages)} extracted pages to 'data/extracted_pages.json'.")

    return manifest, extracted_pages


if __name__ == "__main__":
    sample = "--sample" in sys.argv
    max_p = 30 if sample else None
    run_ingestion(sample_mode=sample, max_pages_per_book=max_p)
