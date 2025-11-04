# PDF to Markdown Converter (Obsidian-Ready)

A powerful, single-file GUI application that converts PDFs to clean, well-formatted Markdown optimized for Obsidian and other note-taking apps.

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## ✨ Features

### Smart Text Processing
- **Intelligent heading detection** - Identifies headings by font size and ALL-CAPS/MOSTLY-CAPS patterns
- **Paragraph reconstruction** - Merges wrapped lines and fixes hyphenation (e.g., "trans-\nform" → "transform")
- **Orphan fragment merging** - Combines short isolated lines into previous paragraphs for better flow
- **Drop cap handling** - Ignores oversized decorative first letters

### Formatting Preservation
- **Bold** and *italic* text when available in the PDF
- Bullet lists (•, ◦, ·, -, —)
- Numbered lists (1. 2. 3.)
- Lettered outlines (a) b) c))

### Automatic Cleanup
- **Header/footer removal** - Auto-detects and strips repeating headers/footers across pages
- **URL linkification** - Converts plain URLs to clickable Markdown links
- **Punctuation normalization** - Fixes smart quotes, ellipses, and em-dashes

### Image Support
- **Export images** to an `_assets/` folder with relative Markdown links
- Automatic conversion to PNG format
- Per-page image organization

### Quality of Life
- **Preview mode** - Test settings on first 3 pages before converting large PDFs
- **Per-page error recovery** - Continues conversion even if individual pages fail
- **Persistent settings** - Remembers your last-used paths and options
- **Progress tracking** - Real-time progress bar showing page-by-page conversion
- **Configurable options** - Fine-tune heading detection, orphan merging, and more

## 📦 Installation

### Option 1: Python (All Platforms)

1. **Install Python 3.7+** if you don't have it already

2. **Install PyMuPDF**:
```bash
pip install pymupdf
```

3. **Download the script**:
```bash
# Clone the repository
git clone https://github.com/yourusername/pdf_to_md.git
cd pdf_to_md

# Or download pdf_to_md.py directly
```

4. **Run the application**:
```bash
python pdf_to_md.py
```

### Option 2: Windows Executable (No Python Required)

1. Download the latest `.exe` from [Releases](https://github.com/yourusername/pdf-to-markdown/releases)
2. Double-click to run - no installation needed!

## 🚀 Usage

### Quick Start

1. **Launch the application**
2. **Browse** for your input PDF file
3. **Choose** where to save the output `.md` file
4. **Configure options** (or use the smart defaults)
5. **Click Convert**

### Options Explained

| Option | Description | Default |
|--------|-------------|---------|
| **Promote ALL-CAPS lines to headings** | Treats lines in all capitals as section headings | ✓ Enabled |
| **Merge short orphan fragments** | Combines isolated short lines into previous paragraphs | ✓ Enabled |
| **Insert page breaks (---)** | Adds horizontal rules between pages | ✗ Disabled |
| **Remove repeating header/footer** | Auto-detects and removes common headers/footers | ✓ Enabled |
| **Export images to _assets/** | Extracts images and creates Markdown links | ✗ Disabled |
| **Preview first 3 pages only** | Test settings on large PDFs without full conversion | ✗ Disabled |
| **Heading size ratio** | Font size multiplier for heading detection (e.g., 1.15) | 1.15 |
| **Orphan max length** | Maximum characters for orphan fragment merging | 45 |

### Example Output

**Input PDF:**
```
CHAPTER 1: INTRODUCTION

This is a paragraph that wraps across
multiple lines in the PDF file.

• First bullet point
• Second bullet point
```

**Output Markdown:**
```markdown
# CHAPTER 1: INTRODUCTION

This is a paragraph that wraps across multiple lines in the PDF file.

- First bullet point
- Second bullet point
```

## 🎯 Use Cases

- **Academic papers** - Convert research PDFs to editable Markdown
- **Books and eBooks** - Extract text for note-taking and annotation
- **Documentation** - Archive PDF manuals as searchable Markdown
- **Obsidian vaults** - Import PDFs directly into your knowledge base
- **Markdown workflows** - Use with any Markdown editor (Typora, VS Code, etc.)

## 🔧 Advanced: Creating a Windows Executable

Want to share this with non-technical users? Create a standalone `.exe`:

```bash
# Install PyInstaller
pip install pyinstaller

# Create the executable
pyinstaller --noconsole --onefile --name "PDF_to_Markdown" pdf_to_md_v2.py

# The .exe will be in the dist/ folder
```

## ⚠️ Limitations

- **Scanned PDFs** - Requires text-based PDFs (not scanned images). For OCR, use Adobe Acrobat or other tools first.
- **Complex tables** - Table formatting may not preserve perfectly
- **Multi-column layouts** - May produce out-of-order text
- **Embedded fonts** - Some proprietary fonts may not render correctly

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- **Report bugs** - Open an issue with details and sample PDFs (if possible)
- **Suggest features** - Share your ideas in the Issues tab
- **Submit pull requests** - Fork, improve, and PR
- **Improve documentation** - Help make this README even better

### Development Setup

```bash
git clone https://github.com/yourusername/pdf-to-markdown.git
cd pdf-to-markdown
pip install pymupdf
python pdf_to_md_v2.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- Designed for seamless integration with [Obsidian](https://obsidian.md/)
- Inspired by the need for better PDF → Markdown conversion tools

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/pdf-to-markdown/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pdf-to-markdown/discussions)

## ⭐ Star History

If you find this tool useful, please consider starring the repository!

---

**Made with ❤️ for the Obsidian and Markdown communities**
