# 📄 Doc Analyzer - AI-Powered PDF Chat Application

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

![Doc Analyzer Demo](assets/demo.gif) <!-- Replace with actual demo media -->

An intelligent document analysis tool powered by Google's Gemini Pro AI that enables natural language Q&A with PDF documents through an intuitive chat interface.

## 🌟 Features

- **Document Processing**
  - PDF text extraction and intelligent chunking
  - Context-aware document analysis
  - Multi-page document support (first 3-4 pages processed for quick responses)

- **AI Capabilities**
  - Google Gemini Pro integration
  - Natural language understanding
  - Contextual question answering
  - Response formatting and sanitization

- **User Interface**
  - Streamlit-based web interface
  - Chat-style interaction with history
  - Responsive design for mobile/desktop
  - File upload progress indicators
  - Error handling and validation

- **Security**
  - API key encryption
  - Session-based document handling
  - Temporary file cleanup
  - Environment variable management

## 📋 Prerequisites

- Python 3.10+
- Google API key ([Get from AI Studio](https://aistudio.google.com/))
- 1GB+ free RAM (for document processing)

## 🚀 Quick Start

### Local Installation

1. Clone repository:
```bash
git clone https://github.com/waqaskhosa99/pdfagent.git
cd doc-analyzer