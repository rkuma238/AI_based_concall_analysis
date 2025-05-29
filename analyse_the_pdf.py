import re
import os
import shutil
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import urllib.parse
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Prompt
from typing import Optional, List, Dict, Any
import traceback
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
import uuid
import time
from openai import OpenAI

# Base URL of the Screener.in page for companies
BASE_URL = "https://www.screener.in/company/{}/consolidated/"
BASE_DIR = "/Users/rakeshkumarr/transcript_analyser"

# OpenRouter API Key
OPENROUTER_API_KEY = ""

# Constants for site information
SITE_URL = "https://financialanalyst.local"
SITE_NAME = "Financial Earnings Analyzer"

# ChromaDB settings
CHROMA_DB_DIR = "./chroma_db"

# Initialize Rich console for better output
console = Console()

def extract_and_download_concall_pdfs(company_code):
    """
    Extracts concall-related PDF links from the webpage and downloads them to a company-specific folder.
    
    :param company_code: The company code to use for the folder name and URL.
    :return: A tuple of (all_pdf_links, latest_pdf_links) as lists of dictionaries.
    """
    # Generate URL using the company code
    url = BASE_URL.format(company_code)
    
    try:
        # Create the company directory, deleting it first if it exists
        company_dir = os.path.join(BASE_DIR, company_code)
        if os.path.exists(company_dir):
            console.print(f"[yellow]Deleting existing directory: {company_dir}[/yellow]")
            shutil.rmtree(company_dir)
            
        # Create the company directory
        os.makedirs(company_dir)
        console.print(f"[green]Created directory: {company_dir}[/green]")
        
        # Set up headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Fetch the webpage content
        console.print(f"[blue]Fetching webpage: {url}[/blue]")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        html_content = response.text
        console.print(f"[green]Successfully fetched HTML content ({len(html_content)} bytes)[/green]")
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Save HTML to a file for inspection if needed
        with open(os.path.join(company_dir, 'debug_webpage.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
            console.print("[green]Saved HTML content to debug_webpage.html for inspection[/green]")
        
        # Count all PDF links for reference
        all_pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$'))
        console.print(f"[blue]Total PDF links on page: {len(all_pdf_links)}[/blue]")
        
        # Try to find the concalls section - look for the section with "Concalls" heading
        # First approach: check all headings
        concalls_heading = None
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            if 'concall' in heading.get_text().lower():
                concalls_heading = heading
                console.print(f"[green]Found concalls heading: {heading.get_text()}[/green]")
                break
        
        concalls_container = None
        pdf_links = []
        latest_pdf_links = []  # New list for latest concall PDF links
        latest_date = None  # Track the latest date
        
        if concalls_heading:
            # Try to find the container that holds concall entries
            # Look for parent elements first
            parent_div = concalls_heading.find_parent('div', class_=lambda c: c and 'documents' in c)
            if parent_div:
                console.print(f"[green]Found parent container with class: {parent_div.get('class')}[/green]")
                # Look for show-more-box - common pattern on screener.in
                concalls_container = parent_div.find('div', class_='show-more-box')
                if concalls_container:
                    console.print("[green]Found show-more-box container[/green]")
                else:
                    # If no show-more-box, use the parent div
                    concalls_container = parent_div
            else:
                # Try the next sibling or next element
                next_elem = concalls_heading.find_next('div')
                if next_elem:
                    concalls_container = next_elem
                    console.print("[green]Found next element after heading as container[/green]")
        else:
            # Second approach: look for divs with concalls in class or id
            concalls_div = soup.find('div', class_=lambda c: c and 'concall' in c.lower())
            if concalls_div:
                concalls_container = concalls_div
                console.print(f"[green]Found concalls div by class: {concalls_div.get('class')}[/green]")
            else:
                # Third approach: look for "Documents" section and then concall subsection
                documents_section = soup.find('section', id='documents')
                if documents_section:
                    console.print("[green]Found documents section[/green]")
                    # Look for concalls within documents
                    concalls_container = documents_section.find(
                        lambda tag: tag.name and tag.string and 'concall' in tag.get_text().lower()
                    )
                    if concalls_container:
                        console.print("[green]Found concalls container within documents section[/green]")
                else:
                    console.print("[yellow]No documents section found with id='documents'[/yellow]")
        
        if not concalls_container:
            # Final approach: look for any list containing concall-like PDFs
            console.print("[blue]Looking for lists with concall-like PDFs...[/blue]")
            for ul in soup.find_all('ul'):
                pdf_links_in_ul = ul.find_all('a', href=re.compile(r'\.pdf$'))
                if pdf_links_in_ul and any('transcript' in link.get_text().lower() or 'concall' in link.get_text().lower() for link in pdf_links_in_ul):
                    concalls_container = ul
                    console.print(f"[green]Found list with concall-like PDFs: {len(pdf_links_in_ul)} links[/green]")
                    break
        
        # Function to parse a date string in "Month YYYY" format to a datetime object
        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, "%b %Y")
            except ValueError:
                return None
        
        # Function to create a valid filename from a document name
        def create_filename(name, date_text, doc_type):
            # Replace spaces with underscores and remove any invalid characters
            base_name = f"{company_code}_{doc_type}_{date_text.replace(' ', '_')}"
            # Remove any special characters that might cause issues in filenames
            base_name = re.sub(r'[^\w\-_.]', '_', base_name)
            return f"{base_name}.pdf"
                
        if concalls_container:
            # Extract entries from the concalls container
            # Look for list items which usually contain the date and links
            concall_entries = concalls_container.find_all('li')
            console.print(f"[blue]Found {len(concall_entries)} potential concall entries[/blue]")
            
            seen_urls = set()  # To avoid duplicates
            date_entries = {}  # Dictionary to group links by date
            
            for entry in concall_entries:
                # Try to find the date label - it's usually in a div with certain classes
                date_label_elem = entry.find(lambda tag: tag.name == 'div' and tag.get('class') and 
                                          any('ink-' in c or 'font-' in c or 'nowrap' in c for c in tag.get('class')))
                
                date_text = "Unknown_Date"
                if date_label_elem:
                    date_text = date_label_elem.get_text(strip=True)
                    console.print(f"[blue]Found date label: {date_text}[/blue]")
                else:
                    # Try a regex approach if no date label div found
                    entry_text = entry.get_text()
                    date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', entry_text)
                    if date_match:
                        date_text = date_match.group(0)
                        console.print(f"[blue]Extracted date via regex: {date_text}[/blue]")
                
                # Find all PDF links in this entry
                links = entry.find_all('a', href=re.compile(r'\.pdf$'))
                
                if not links:
                    links = entry.find_all('a', class_='concall-link')
                    console.print(f"[yellow]No PDF links found in entry. Found {len(links)} concall-links instead.[/yellow]")
                    
                for link in links:
                    if not link.has_attr('href') or not link['href'].endswith('.pdf'):
                        continue
                        
                    pdf_url = link['href']
                    
                    # Skip duplicates
                    if pdf_url in seen_urls:
                        continue
                    seen_urls.add(pdf_url)
                    
                    link_text = link.get_text(strip=True)
                    console.print(f"[blue]Found link: {link_text} - {pdf_url}[/blue]")
                    
                    # Determine document type
                    if 'transcript' in link_text.lower() or 'transcript' in pdf_url.lower():
                        doc_type = "Transcript"
                    elif 'ppt' in link_text.lower() or 'presentation' in link_text.lower() or 'ppt' in pdf_url.lower():
                        doc_type = "Presentation"
                    else:
                        doc_type = link_text if link_text else "Document"
                    
                    # Create proper filename (replace spaces with underscores, etc.)
                    filename = create_filename(link_text, date_text, doc_type)
                    file_path = os.path.join(company_dir, filename)
                    
                    # Download the PDF
                    try:
                        console.print(f"[blue]Downloading PDF from {pdf_url} to {file_path}...[/blue]")
                        pdf_response = requests.get(pdf_url, headers=headers)
                        pdf_response.raise_for_status()
                        
                        with open(file_path, 'wb') as f:
                            f.write(pdf_response.content)
                        console.print(f"[green]Successfully downloaded PDF to {file_path}[/green]")
                        
                        # Store local file path in the link info
                        local_path = file_path
                    except Exception as e:
                        console.print(f"[red]Error downloading PDF: {e}[/red]")
                        local_path = None
                    
                    name = f"{doc_type} - {date_text}"
                    context = entry.get_text(separator=" ", strip=True)
                    
                    pdf_link = {
                        "name": name,
                        "url": pdf_url,
                        "link_text": link_text,
                        "context": context,
                        "date_text": date_text,
                        "date": parse_date(date_text),
                        "local_path": local_path,
                        "filename": filename,
                        "doc_type": doc_type
                    }
                    
                    pdf_links.append(pdf_link)
                    
                    # Add to date_entries dictionary for later sorting
                    if date_text not in date_entries:
                        date_entries[date_text] = []
                    date_entries[date_text].append(pdf_link)
                    
                    console.print(f"[green]Added: {name} - {pdf_url}[/green]")
        else:
            # If no container found, look for PDF links with concall/transcript keywords
            console.print("[yellow]No concalls container found, searching for PDF links with relevant keywords...[/yellow]")
            for link in all_pdf_links:
                pdf_url = link['href']
                link_text = link.get_text(strip=True).lower()
                
                # Check if this looks like a concall document
                if any(keyword in link_text or keyword in pdf_url.lower() 
                       for keyword in ['transcript', 'concall', 'con call', 'earning', 'result']):
                    
                    if pdf_url in [p['url'] for p in pdf_links]:
                        continue  # Skip if already added
                    
                    # Try to find date context
                    parent = link.parent
                    date_text = "Unknown_Date"
                    
                    # Look in current element and parent elements for date
                    for i in range(3):  # Check up to 3 levels up
                        if not parent:
                            break
                        text_content = parent.get_text()
                        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', text_content)
                        if date_match:
                            date_text = date_match.group(0)
                            break
                        parent = parent.parent
                    
                    # Determine doc type from link text
                    if 'transcript' in link_text:
                        doc_type = "Transcript"
                    elif 'ppt' in link_text or 'presentation' in link_text:
                        doc_type = "Presentation"
                    else:
                        doc_type = "Concall_Document"
                    
                    # Create filename
                    filename = create_filename(link.get_text(strip=True), date_text, doc_type)
                    file_path = os.path.join(company_dir, filename)
                    
                    # Download the PDF
                    try:
                        console.print(f"[blue]Downloading PDF from {pdf_url} to {file_path}...[/blue]")
                        pdf_response = requests.get(pdf_url, headers=headers)
                        pdf_response.raise_for_status()
                        
                        with open(file_path, 'wb') as f:
                            f.write(pdf_response.content)
                        console.print(f"[green]Successfully downloaded PDF to {file_path}[/green]")
                        
                        # Store local file path in the link info
                        local_path = file_path
                    except Exception as e:
                        console.print(f"[red]Error downloading PDF: {e}[/red]")
                        local_path = None
                    
                    name = f"{doc_type} - {date_text}"
                    context = link.get_text(strip=True)
                    if parent:
                        context = parent.get_text(separator=" ", strip=True)
                    
                    pdf_link = {
                        "name": name,
                        "url": pdf_url,
                        "link_text": link.get_text(strip=True),
                        "context": context,
                        "date_text": date_text,
                        "date": parse_date(date_text),
                        "local_path": local_path,
                        "filename": filename,
                        "doc_type": doc_type
                    }
                    
                    pdf_links.append(pdf_link)
                    
                    # Add to date_entries dictionary
                    if date_text not in date_entries:
                        date_entries[date_text] = []
                    date_entries[date_text].append(pdf_link)
                    
                    console.print(f"[green]Added: {name} - {pdf_url}[/green]")
        
        # Find the latest date
        latest_date_text = None
        latest_datetime = None
        
        for date_text, date_links in date_entries.items():
            parsed_date = parse_date(date_text)
            if parsed_date and (latest_datetime is None or parsed_date > latest_datetime):
                latest_datetime = parsed_date
                latest_date_text = date_text
        
        # Get the latest PDF links
        if latest_date_text:
            console.print(f"\n[green]Latest concall date found: {latest_date_text}[/green]")
            latest_pdf_links = date_entries[latest_date_text]
            console.print(f"[blue]Found {len(latest_pdf_links)} PDF links for the latest date[/blue]")
        
        if not pdf_links:
            console.print("[yellow]No concall-related PDF links found on the webpage.[/yellow]")
            
        return pdf_links, latest_pdf_links
        
    except Exception as e:
        console.print(f"[red]Error extracting and downloading concall PDFs: {e}[/red]")
        traceback.print_exc()
        return [], []

def extract_text_from_pdf(pdf_path):
    """Extract text content from a local PDF file"""
    try:
        # Open and read the local PDF file
        reader = PyPDF2.PdfReader(pdf_path)
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting text from PDF...", total=len(reader.pages))
            
            text = ""
            for i, page in enumerate(reader.pages):
                extracted_text = page.extract_text() or ""
                text += extracted_text + "\n\n"
                progress.update(task, advance=1)
        
        return text
    except Exception as e:
        console.print(f"[red]Error extracting text from PDF: {str(e)}[/red]")
        return ""

def setup_chromadb(company_code):
    """Set up and connect to ChromaDB with appropriate embeddings model"""
    collection_name = f"{company_code}_earnings"
    
    try:
        # Create client
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Set up sentence-transformers embedding function - good for financial text
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Get existing collection
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            console.print(f"[green]Connected to existing collection: {collection_name}[/green]")
        except:
            # Create new collection if it doesn't exist
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            console.print(f"[green]Created new collection: {collection_name}[/green]")
        
        return client, collection
    except Exception as e:
        console.print(f"[red]Error setting up ChromaDB: {str(e)}[/red]")
        return None, None

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for vector storage"""
    chunks = []
    
    if not text:
        return chunks
    
    # For financial documents, try to split on paragraph boundaries when possible
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph exceeds chunk size, store current chunk and start new one
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text
            
        current_chunk += " " + paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If any chunks are too small, combine them
    min_chunk_size = 200  # Minimum meaningful chunk size
    filtered_chunks = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < min_chunk_size and i < len(chunks) - 1:
            # Combine with next chunk
            chunks[i+1] = chunk + " " + chunks[i+1]
        else:
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def load_pdf_into_chromadb(pdf_path, collection):
    """Load PDF content into ChromaDB"""
    try:
        # Extract text from PDF
        console.print(f"[cyan]Processing: {os.path.basename(pdf_path)}[/cyan]")
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            console.print("[red]Error: Could not extract text from PDF[/red]")
            return False, None
        
        console.print(f"[green]Successfully extracted {len(text)} characters[/green]")
        
        # Split text into chunks
        console.print("[cyan]Splitting text into semantic chunks...[/cyan]")
        chunks = split_text_into_chunks(text)
        console.print(f"[green]Created {len(chunks)} text chunks[/green]")
        
        # First, delete any existing documents if we're reloading the same PDF
        try:
            existing_ids = collection.get()['ids']
            if existing_ids:
                console.print(f"[yellow]Removing {len(existing_ids)} existing documents from collection[/yellow]")
                collection.delete(existing_ids)
        except Exception as e:
            console.print(f"[yellow]No existing documents to remove: {str(e)}[/yellow]")
        
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Add metadata for better retrieval
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": os.path.basename(pdf_path),
                "chunk_id": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk)
            })
        
        # Insert chunks into ChromaDB
        console.print("[cyan]Adding text chunks to ChromaDB...[/cyan]")
        
        # Add in batches to prevent memory issues with large documents
        batch_size = 20
        with Progress() as progress:
            task = progress.add_task("[cyan]Adding chunks to database...", total=len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                collection.add(
                    documents=chunks[i:end_idx],
                    ids=ids[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                progress.update(task, advance=end_idx - i)
        
        console.print(f"[green]Successfully loaded {len(chunks)} chunks into ChromaDB[/green]")
        return True, text
        
    except Exception as e:
        console.print(f"[red]Error loading PDF into ChromaDB: {str(e)}[/red]")
        traceback.print_exc()
        return False, None

def get_relevant_context(query, collection, n_results=10):
    """Retrieve relevant context from ChromaDB based on query"""
    try:
        # Enhance the query with financial context if needed
        financial_keywords = ["revenue", "profit", "earnings", "margin", "growth", 
                             "q1", "q2", "q3", "q4", "quarter", "fiscal", "year",
                             "dividend", "eps", "ebitda", "net income", "cash flow",
                             "balance sheet", "income statement", "guidance"]
                             
        enhanced_query = query
        
        # Check if query is about financials but doesn't use specific terms
        for keyword in financial_keywords:
            if keyword.lower() in query.lower():
                # Query already has financial terms
                break
        else:
            # If no financial terms found, check for general financial questions
            if any(term in query.lower() for term in ["how much", "financial", "perform", "result", "number"]):
                enhanced_query = f"financial results {query}"
        
        # Debug information
        console.print(f"[dim]Using query: {enhanced_query}[/dim]")
        
        # Perform the query - increase n_results to get more potential matches
        results = collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Debug information about results
        if not results or 'documents' not in results:
            console.print("[red]No results returned from ChromaDB query[/red]")
            return ""
            
        if len(results['documents']) == 0 or len(results['documents'][0]) == 0:
            console.print("[red]Empty results returned from ChromaDB query[/red]")
            return ""
            
        console.print(f"[dim]Found {len(results['documents'][0])} potential context chunks[/dim]")
        
        # Get results
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [{}] * len(documents)
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(documents)
        
        # Lower the relevance threshold to include more results
        relevance_threshold = 0.0  # Accept all results initially
        
        # Combine results without filtering by relevance initially
        combined_results = []
        for i in range(len(documents)):
            doc = documents[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            
            # Skip empty documents
            if not doc or len(doc.strip()) < 20:
                continue
                
            relevance = 1.0 - min(distance, 1.0)  # Convert distance to relevance (0-1)
            combined_results.append((doc, metadata, relevance))
        
        # Sort by relevance
        combined_results.sort(key=lambda x: x[2], reverse=True)
        
        # If we have too many results, just take the top ones
        max_chunks = 5  # Limit to top chunks to avoid token limits
        if len(combined_results) > max_chunks:
            combined_results = combined_results[:max_chunks]
            
        # Join the relevant chunks
        context_parts = []
        for doc, metadata, relevance in combined_results:
            chunk_info = f"Relevance: {relevance:.2f}"
            console.print(f"[dim]Using chunk with {len(doc)} chars and relevance {relevance:.2f}[/dim]")
            context_parts.append(f"{doc}")
        
        full_context = "\n\n".join(context_parts)
        
        # Limit total context size to avoid token limits
        max_context_length = 15000
        if len(full_context) > max_context_length:
            console.print(f"[yellow]Truncating context from {len(full_context)} to {max_context_length} chars[/yellow]")
            full_context = full_context[:max_context_length]
            
        return full_context
    except Exception as e:
        console.print(f"[red]Error querying ChromaDB: {str(e)}[/red]")
        traceback.print_exc()
        return ""

def analyze_with_llm(query, context, company_code="Unknown"):
    """Analyze context using OpenRouter LLM, optimized for financial analysis"""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    if not context or len(context) < 100:
        return "Insufficient relevant information found in the document to answer this query."
    
    # Create a prompt that instructs the LLM to analyze based on the retrieved context
    prompt = f"""
You are an expert financial analyst tasked with analyzing information for {company_code}.
Please analyze the following excerpts from a financial document and answer this specific question:

QUESTION: {query}

Use only the information in the PROVIDED CONTEXT below. If the information needed is not in the context, 
state "This information is not available in the provided context."

PROVIDED CONTEXT:
{context}

Important guidelines:
- You are analyzing a financial document, likely an earnings call transcript or investor presentation
- Only provide information that is explicitly mentioned in the context
- For financial questions, cite specific numbers, percentages, and metrics with their exact values
- If financial metrics are found, specify the time period they refer to (e.g., Q1 2025, FY2024, etc.)
- If trends are mentioned, note whether they are increasing or decreasing and by how much
- Compare current figures to previous periods if that information is available
- Provide a clear, concise answer focused specifically on the question
- If the answer requires financial expertise to interpret the data, provide that expert interpretation
"""
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for more factual responses
            max_tokens=4000,   # Adjust based on your needs
        )
        
        analysis = completion.choices[0].message.content
        return analysis
    except Exception as e:
        console.print(f"[red]Error generating analysis: {str(e)}[/red]")
        return f"Error generating analysis: {str(e)}"

def analyze_full_text_with_llm(question, full_text, company_code="Unknown"):
    """
    Analyze the full text directly with LLM (non-RAG approach)
    This uses the entire document text for context instead of retrieval-based chunks
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    if not full_text or len(full_text) < 100:
        return "Insufficient text content to analyze."
    
    # Truncate text if it's too long to fit in context window
    max_text_length = 40000  # Adjust based on model's context limits
    if len(full_text) > max_text_length:
        truncated_text = full_text[:max_text_length]
        console.print(f"[yellow]Text is too long ({len(full_text)} chars), truncating to {max_text_length} chars[/yellow]")
    else:
        truncated_text = full_text
    
    # Create a prompt for direct analysis
    prompt = f"""
You are an expert financial analyst tasked with analyzing an earnings call transcript for {company_code}.
Please analyze the following transcript and answer this specific question:

QUESTION: {question}

TRANSCRIPT:
{truncated_text}

Important guidelines:
- You are analyzing an earnings call transcript
- Only provide information that is explicitly mentioned in the transcript
- For financial questions, cite specific numbers, percentages, and metrics with their exact values
- If financial metrics are found, specify the time period they refer to (e.g., Q1 2025, FY2024, etc.)
- If trends are mentioned, note whether they are increasing or decreasing and by how much
- Compare current figures to previous periods if that information is available
- Provide a clear, concise answer focused specifically on the question
- If the information isn't available in the transcript, clearly state this fact
"""
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            model="meta-llama/llama-3.3-8b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        
        analysis = completion.choices[0].message.content
        return analysis
    except Exception as e:
        console.print(f"[red]Error generating analysis: {str(e)}[/red]")
        return f"Error generating analysis: {str(e)}"

def run_comprehensive_analysis(collection, pdf_path, company_code, use_rag=True, full_text=None):
    """
    Run a comprehensive financial analysis on the document
    
    :param collection: ChromaDB collection for RAG-based retrieval
    :param pdf_path: Path to the PDF file being analyzed
    :param company_code: Company code for the analysis
    :param use_rag: Whether to use RAG (True) or full-text analysis (False)
    :param full_text: Full text of the document if already extracted
    """
    console.print(Panel.fit(
        f"[bold green]Automated Financial Analysis[/bold green]\n\n"
        f"Company: [bold]{company_code}[/bold]\n"
        f"Method: [bold]{'RAG-based analysis' if use_rag else 'Full-text analysis'}[/bold]\n"
        f"Running comprehensive financial analysis...",
        title="Analysis In Progress",
        border_style="cyan"
    ))
    
    # If we don't have the full text yet and we're not using RAG, extract it
    if not use_rag and not full_text:
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text or len(full_text) < 100:
            console.print("[red]Error: Could not extract sufficient text from PDF for analysis[/red]")
            return
    
    # Define key analysis questions
    analysis_questions = [
        "What are the key financial highlights and performance metrics mentioned in this document?",
        "What are the company's revenue figures and how have they changed compared to previous periods?",
        "What are the profit margins and earnings figures mentioned in the document?",
        "What is the company's outlook or guidance for future periods?",
        "What are the key business segments and their performance?",
        "What are the main challenges or risks mentioned in the document?",
        "What strategic initiatives or growth plans are discussed in the document?",
        "What information is provided about the company's cash flow, debt, and balance sheet?",
        "What information is provided about dividends, share buybacks, or other shareholder returns?",
        "What are the key operational metrics mentioned in the document?",
        "Analyze any information about the order book, backlog, or upcoming project executions."
    ]
    
    # Run analysis for each question
    results = {}
    
    for i, question in enumerate(analysis_questions):
        console.print(f"[cyan]Analysis section {i+1}/{len(analysis_questions)}:[/cyan] [bold]{question}[/bold]")
        
        if use_rag:
            # RAG-based approach - get relevant context first
            with console.status("[cyan]Retrieving relevant financial information...[/cyan]"):
                context = get_relevant_context(question, collection)
            
            if not context or len(context) < 100:
                console.print("[yellow]Insufficient relevant information found for this section.[/yellow]")
                results[question] = "Insufficient information available in the document."
                continue
                
            # Analyze with LLM using retrieved context
            with console.status("[cyan]Analyzing financial data...[/cyan]"):
                analysis = analyze_with_llm(question, context, company_code)
                results[question] = analysis
        else:
            # Non-RAG approach - use full text directly
            with console.status("[cyan]Analyzing full transcript...[/cyan]"):
                analysis = analyze_full_text_with_llm(question, full_text, company_code)
                results[question] = analysis
        
        # Add a brief pause to avoid rate limiting
        time.sleep(1)
    
    # Generate executive summary
    console.print("[cyan]Generating executive summary...[/cyan]")
    
    # Create a summary of all the analysis parts
    summary_prompt = f"""
As a senior financial analyst, create a concise executive summary of the following financial analysis for {company_code}.
Focus on the most important financial metrics, business performance, and outlook.
Highlight any significant changes, trends, or strategic initiatives.
Keep the summary factual and evidence-based, using only information explicitly provided in the analysis below.

ANALYSIS SECTIONS:
"""
    
    for i, question in enumerate(analysis_questions):
        if results[question] != "Insufficient information available in the document.":
            summary_prompt += f"\n{i+1}. {question}\n{results[question]}\n"
    
    with console.status("[cyan]Creating executive summary...[/cyan]"):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                },
                model="meta-llama/llama-3.3-8b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            
            executive_summary = completion.choices[0].message.content
        except Exception as e:
            executive_summary = f"Error generating executive summary: {str(e)}"
    
    # Save analysis to file
    analysis_filename = f"{company_code}_{'rag' if use_rag else 'full'}_analysis.md"
    analysis_path = os.path.join(BASE_DIR, company_code, analysis_filename)
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"# {company_code} Financial Analysis\n\n")
        f.write(f"Analysis of: {os.path.basename(pdf_path)}\n")
        f.write(f"Method: {'RAG-based analysis' if use_rag else 'Full-text analysis'}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(executive_summary + "\n\n")
        
        f.write("## Detailed Analysis\n\n")
        for i, question in enumerate(analysis_questions):
            f.write(f"### {question}\n\n")
            f.write(results[question] + "\n\n")
    
    console.print(f"[green]Analysis saved to: {analysis_path}[/green]")
    
    # Print the complete analysis
    console.print("\n")
    console.print(Panel.fit(
        executive_summary,
        title="Executive Summary",
        border_style="green"
    ))
    
    console.print("\n[bold cyan]Detailed Analysis Sections:[/bold cyan]")
    
    for i, question in enumerate(analysis_questions):
        console.print(Panel.fit(
            results[question],
            title=f"{i+1}. {question}",
            border_style="blue"
        ))
        console.print("\n")
    
    return analysis_path

def main(company_code: str = "CCL", use_rag: bool = True):
    """
    Main entry point for the application
    
    :param company_code: The company code to analyze (e.g., "AARTIPHARM")
    :param use_rag: Whether to use RAG for analysis (True) or full-text analysis (False)
    """
    console.print("\n[bold cyan]Transcript Analyzer for Financial Earnings Calls[/bold cyan]")
    console.print(f"Company: [bold]{company_code}[/bold], Analysis Method: [bold]{'RAG' if use_rag else 'Full-text'}[/bold]")
    
    # Step 1: Download the latest concall PDFs
    console.print("\n[bold blue]Step 1: Downloading latest concall PDFs[/bold blue]")
    all_pdfs, latest_pdfs = extract_and_download_concall_pdfs(company_code)
    
    if not latest_pdfs:
        console.print("[red]No concall PDF files found for the latest period. Exiting.[/red]")
        return
    
    # Step 2: Find the transcript PDF from the latest concalls
    console.print("\n[bold blue]Step 2: Finding transcript PDF from latest concalls[/bold blue]")
    transcript_pdfs = [pdf for pdf in latest_pdfs if pdf['doc_type'] == 'Transcript']
    
    if not transcript_pdfs:
        console.print("[yellow]No transcript PDF found in the latest concall documents.[/yellow]")
        console.print("[yellow]Looking for any other document type to analyze...[/yellow]")
        
        # If no transcript, try to use any PDF we have
        if latest_pdfs:
            transcript_pdfs = latest_pdfs  # Use whatever we have
        else:
            console.print("[red]No suitable documents found for analysis. Exiting.[/red]")
            return
    
    # Step 3: Select the first transcript PDF to analyze
    transcript_pdf = transcript_pdfs[0]
    console.print(f"[green]Selected document for analysis: {transcript_pdf['name']}[/green]")
    
    pdf_path = transcript_pdf['local_path']
    if not pdf_path or not os.path.exists(pdf_path):
        console.print(f"[red]PDF file not found at {pdf_path}. Exiting.[/red]")
        return
    
    # Get company name from PDF file
    company_name = company_code
    
    # For RAG analysis, set up ChromaDB and load the PDF
    full_text = None
    collection = None
    
    if use_rag:
        # Step 4a: Set up ChromaDB for RAG-based analysis
        console.print("\n[bold blue]Step 4: Setting up vector database for RAG analysis[/bold blue]")
        _, collection = setup_chromadb(company_code)
        
        if not collection:
            console.print("[red]Failed to set up ChromaDB. Falling back to full-text analysis.[/red]")
            use_rag = False
        else:
            # Step 5a: Load the PDF into ChromaDB
            console.print("\n[bold blue]Step 5: Loading PDF into vector database[/bold blue]")
            success, full_text = load_pdf_into_chromadb(pdf_path, collection)
            
            if not success:
                console.print("[red]Failed to load PDF into ChromaDB. Falling back to full-text analysis.[/red]")
                use_rag = False
    else:
        # Step 4b-5b: Extract full text for non-RAG analysis
        console.print("\n[bold blue]Step 4-5: Extracting full text for direct analysis[/bold blue]")
        full_text = extract_text_from_pdf(pdf_path)
        
        if not full_text or len(full_text) < 100:
            console.print("[red]Failed to extract text from PDF. Exiting.[/red]")
            return
    
    # Step 6: Run comprehensive analysis
    console.print("\n[bold blue]Step 6: Running comprehensive financial analysis[/bold blue]")
    analysis_path = run_comprehensive_analysis(
        collection, 
        pdf_path, 
        company_code, 
        use_rag=use_rag, 
        full_text=full_text
    )
    
    console.print(f"\n[bold green]Analysis completed successfully![/bold green]")
    console.print(f"Analysis saved to: {analysis_path}")

if __name__ == "__main__":
    typer.run(main)
