# 📧 Business Email PDF Analysis & Recommendation System

> **Project Presentation (PPT)** : [View Presentation Slides](https://docs.google.com/presentation/d/10cDAfyUDkmXcXXc1y-ilhWFxBKG5liBP/edit?usp=drive_link&ouid=117197231862834975531&rtpof=true&sd=true)
> 
> **Demo Video**  : [Watch Demo Video](https://drive.google.com/file/d/1rtUoST3fyLVT2XfT4RQPpR-zUCX6546Y/view?usp=drive_link)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Airflow](https://img.shields.io/badge/Apache-Airflow-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)
![Gemini](https://img.shields.io/badge/AI-Gemini-purple)

## Overview

This project is an automated business email analysis and recommendation system built with Apache Airflow, Google Gemini AI, Streamlit, and Chroma DB.

The system automatically collects business emails, extracts text from attached PDF files, summarizes the content using Gemini AI, stores the results as vector embeddings, and recommends similar historical emails based on semantic similarity.

---

## Key Features

* Automated email collection pipeline using Apache Airflow
* PDF text extraction from email attachments
* AI-powered email and PDF summarization with Google Gemini
* TF-IDF keyword visualization in Streamlit
* Similar email recommendation using Cosine Similarity
* Vector-based semantic search with Chroma DB
* Dedicated Chroma DB viewer for inspecting stored email data

---

## Tech Stack

### Language & Package Management

* Python 3.12
* uv

### Dashboard

* Streamlit

### Data Pipeline

* Apache Airflow
* Docker Compose

### Database

* Chroma DB
* PostgreSQL

### AI & Machine Learning

* Google Gemini API

  * gemini-1.5-flash for summarization
  * text-embedding-004 for embeddings
* scikit-learn

  * TF-IDF
  * Cosine Similarity

---

## Prerequisites

Before running this project, make sure the following tools are installed:

* Docker
* Docker Compose
* uv

Create a `.env` file in the project root directory.

```ini
GEMINI_API_KEY=your_gemini_api_key
IMAP_SERVER=imap.gmail.com
EMAIL_ACCOUNT=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

---

## How to Run

### 1. Start Infrastructure

Run Airflow, PostgreSQL, and Chroma DB with Docker Compose.

```bash
docker compose up -d
```

After the containers are running, open the Airflow UI:

```text
http://localhost:8080
```

Default Airflow account:

```text
ID: admin
Password: admin
```

Enable or manually trigger the DAG:

```text
email_embedding_daily_job
```

---

### 2. Install Dependencies

```bash
uv sync
```

---

### 3. Run Main Streamlit App

```bash
uv run streamlit run src/app.py
```

Open the dashboard:

```text
http://localhost:8501
```

---

### 4. Run Chroma DB Viewer

```bash
uv run streamlit run scripts/db_viewer.py
```

Open the DB viewer:

```text
http://localhost:8502
```

---

## Project Structure

```text
.
├── dags/
│   └── email_embedding_pipeline.py
├── src/
│   ├── app.py
│   ├── config.py
│   ├── db_client.py
│   ├── email_client.py
│   ├── pdf_parser.py
│   ├── summarizer.py
│   └── word_cloud_gen.py
├── scripts/
│   └── db_viewer.py
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Expected Impact

* Reduces repetitive email and PDF review time
* Converts personal inbox data into reusable knowledge assets
* Enables semantic search beyond simple keyword matching
* Recommends similar historical business cases
* Supports faster decision-making and knowledge reuse

---

## Vision

This project aims to transform fragmented business emails into a unified organizational knowledge asset by integrating email ingestion, PDF extraction, LLM summarization, vector embedding, and semantic retrieval into a single automated pipeline.
