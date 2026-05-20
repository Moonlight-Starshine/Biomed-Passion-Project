# Exploring Biomedical Engineering: A Passion-Project

## Description (Original)

A low-cost Anemia Detection AI tool that attaches to any home or communal microscope and:
- automatically counts cells
- measures cell size
- flags abnormalities for diagnosis
- includes a confidence score of the model's prediction

### Why this is impactful
1 in 4 people worldwide struggle with anemia, meaning 1.9 billion people. 1 in 3 of those with anemia are undiagnosed- making simple methods for anemia screening imperative.

But automated cell counters cost $1,000–$6,000. This anemia diagnostic app can be used to diagnose with a phone and a $20 microscope.

### Deliverables
- A simple web app that counts cells from an uploaded photo
- A printable "attachable guide" showing how to align a phone with the microscope

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Usage

Upload a microscope image, and get the cell count, sizes, and report.

## Guide

See [guide/attachable_guide.md](guide/attachable_guide.md) for printable instructions.

## Future Updates
- Considering expanding impact by also integrating detection for:
    - infection monitoring (WBC counts)
    - yeast viability for biotechnology
    - water contamination checks# ESP32
