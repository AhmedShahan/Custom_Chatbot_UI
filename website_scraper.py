import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import time
import pandas as pd
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class WebsiteScraper:
    def __init__(self, start_url, output_folder="scraped_content"):
        self.start_url = start_url
        self.base_url = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
        self.output_folder = output_folder
        self.visited_urls = set()
        self.pages_content = []
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def get_page_title(self, soup):
        """Extract the title of the page"""
        if soup.title:
            return soup.title.text.strip()
        return "Untitled Page"
    
    def clean_text(self, text):
        """Clean text by removing extra whitespace and normalizing Unicode characters"""
        if not text:
            return ""
        # Replace Unicode whitespace variants with standard space
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()
    
    def extract_content(self, soup):
        """Extract text and tables from the page"""
        content = []
        
        # Extract text paragraphs
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for p in paragraphs:
            text = self.clean_text(p.get_text())
            if text:  # Only add non-empty content
                content.append({"type": "text", "content": text, "tag": p.name})
        
        # Extract tables
        tables = soup.find_all('table')
        for idx, table in enumerate(tables):
            rows = []
            for tr in table.find_all('tr'):
                row = [self.clean_text(td.get_text()) for td in tr.find_all(['td', 'th'])]
                if any(cell for cell in row):  # Only add non-empty rows
                    rows.append(row)
            
            if rows:  # Only add non-empty tables
                content.append({"type": "table", "content": rows})
        
        return content
    
    def get_internal_links(self, soup):
        """Find all internal links on the page"""
        internal_links = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Make URL absolute
            if not href.startswith(('http://', 'https://')):
                href = urljoin(self.base_url, href)
            
            # Check if it's an internal link
            if href.startswith(self.base_url):
                # Remove fragments and queries
                href = href.split('#')[0].split('?')[0]
                
                # Remove trailing slash for consistency
                if href.endswith('/'):
                    href = href[:-1]
                
                internal_links.add(href)
        
        return internal_links
    
    def scrape_page(self, url):
        """Scrape a single page"""
        print(f"Scraping: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = self.get_page_title(soup)
                content = self.extract_content(soup)
                
                if content:
                    self.pages_content.append({
                        "url": url,
                        "title": title,
                        "content": content
                    })
                
                return self.get_internal_links(soup)
            else:
                print(f"Failed to retrieve {url}: Status code {response.status_code}")
                return set()
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return set()
    
    def scrape_website(self, max_pages=50):
        """Scrape the website with a page limit for safety"""
        queue = [self.start_url]
        page_count = 0
        
        print(f"\nStarting to scrape the website. Limited to {max_pages} pages for safety.")
        print("The script will continue to run until the limit is reached or all pages are scraped.")
        
        try:
            while queue and page_count < max_pages:
                current_url = queue.pop(0)
                
                if current_url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(current_url)
                page_count += 1
                
                if page_count % 5 == 0:
                    print(f"Progress: {page_count}/{max_pages} pages scraped, {len(queue)} pages in queue")
                
                new_links = self.scrape_page(current_url)
                
                # Add new links to the queue
                for link in new_links:
                    if link not in self.visited_urls and link not in queue:
                        queue.append(link)
                
                # Be nice to the server
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nScraping interrupted by user. Proceeding to generate PDF with scraped content.")
        
        print(f"\nScraping completed. Total pages scraped: {page_count}")
    
    def convert_to_pdf(self, output_filename="website_content.pdf"):
        """Convert all scraped content to PDF using ReportLab (better Unicode support)"""
        if not self.pages_content:
            print("No content to convert to PDF")
            return None
        
        print(f"\nGenerating PDF with content from {len(self.pages_content)} pages...")
        
        pdf_path = os.path.join(self.output_folder, output_filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        url_style = styles['Italic']
        heading_styles = {
            'h1': styles['Heading1'],
            'h2': styles['Heading2'],
            'h3': styles['Heading3'],
            'h4': styles['Heading4'],
            'h5': styles['Heading5'],
            'h6': styles['Heading6'],
            'p': styles['Normal']
        }
        
        # Create content elements
        elements = []
        
        # Add website title
        elements.append(Paragraph(f"Scraped Content from {urlparse(self.start_url).netloc}", title_style))
        elements.append(Spacer(1, 12))
        
        # Process each page
        for page in self.pages_content:
            # Add page title
            elements.append(Paragraph(page["title"], styles['Heading1']))
            elements.append(Paragraph(f"URL: {page['url']}", url_style))
            elements.append(Spacer(1, 12))
            
            # Add page content
            for item in page["content"]:
                if item["type"] == "text":
                    if 'tag' in item and item['tag'] in heading_styles:
                        style = heading_styles[item['tag']]
                    else:
                        style = styles['Normal']
                    elements.append(Paragraph(item["content"], style))
                    elements.append(Spacer(1, 6))
                
                elif item["type"] == "table":
                    if not item["content"]:
                        continue
                        
                    # Create the table
                    table_data = item["content"]
                    table = Table(table_data)
                    
                    # Style the table
                    style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ])
                    table.setStyle(style)
                    elements.append(table)
                    elements.append(Spacer(1, 12))
            
            # Add a separator between pages
            elements.append(Paragraph("=" * 50, styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Build the PDF
        doc.build(elements)
        print(f"PDF saved to {pdf_path}")
        return pdf_path

if __name__ == "__main__":
    website_url = input("Enter website URL to scrape: ")
    output_filename = input("Enter output PDF filename (default: website_content.pdf): ") or "website_content.pdf"
    
    scraper = WebsiteScraper(website_url)
    scraper.scrape_website()
    pdf_path = scraper.convert_to_pdf(output_filename)
    
    print(f"Process completed! PDF saved to: {pdf_path}") 