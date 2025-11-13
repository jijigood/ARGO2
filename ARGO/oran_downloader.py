#!/usr/bin/env python3
"""
O-RAN Specifications Bulk Downloader
Downloads all O-RAN specification documents from specifications.o-ran.org
"""

import requests
import time
import os
import json
import re
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Configuration
DEFAULT_START_ID = 1
DEFAULT_END_ID = 1200
DEFAULT_WORKERS = 3
DEFAULT_OUTPUT_DIR = "oran_specs"
DEFAULT_DELAY = 1.0

class ORANDownloader:
    def __init__(self, output_dir=DEFAULT_OUTPUT_DIR, max_workers=DEFAULT_WORKERS, delay=DEFAULT_DELAY):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.delay = delay
        self.session = requests.Session()
        self.successful = []
        self.failed = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_file(self, doc_id):
        """Download a single file by ID"""
        url = f"https://specifications.o-ran.org/download?id={doc_id}"
        
        try:
            response = self.session.get(
                url, 
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30,
                stream=True
            )
            
            if response.status_code == 200:
                # Get filename from Content-Disposition header
                filename = f"oran_spec_{doc_id}.pdf"
                if 'content-disposition' in response.headers:
                    cd = response.headers['content-disposition']
                    if 'filename=' in cd:
                        filename = cd.split('filename=')[-1].strip('"')
                
                filepath = os.path.join(self.output_dir, filename)
                
                # Download the file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return doc_id, True, filename
            else:
                return doc_id, False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return doc_id, False, str(e)
    
    def download_range(self, start_id, end_id):
        """Download documents in the specified ID range"""
        print(f"\n{'='*60}")
        print(f"O-RAN Specifications Downloader")
        print(f"{'='*60}")
        print(f"Downloading IDs from {start_id} to {end_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"{'='*60}\n")
        
        ids = list(range(start_id, end_id + 1))
        total = len(ids)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_file, doc_id): doc_id for doc_id in ids}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                doc_id, success, result = future.result()
                
                # Progress indicator
                progress = completed / total * 100
                status = "✓" if success else "✗"
                
                if success:
                    self.successful.append((doc_id, result))
                    print(f"[{completed}/{total} - {progress:.1f}%] {status} ID {doc_id}: {result}")
                else:
                    self.failed.append((doc_id, result))
                    if "404" not in result:  # Only show non-404 errors
                        print(f"[{completed}/{total} - {progress:.1f}%] {status} ID {doc_id}: {result}")
                
                time.sleep(self.delay)
        
        return self.successful, self.failed
    
    def organize_files(self):
        """Organize downloaded files by Working Group"""
        print(f"\n{'='*60}")
        print("Organizing files by Working Group...")
        print(f"{'='*60}")
        
        organized_dir = f"{self.output_dir}_organized"
        os.makedirs(organized_dir, exist_ok=True)
        
        # Pattern to match O-RAN naming conventions
        pattern = re.compile(r'O-RAN[.-]?(WG\d+|FG\d+|TIFG)[.-]?')
        
        files_by_group = defaultdict(list)
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.pdf'):
                match = pattern.search(filename)
                if match:
                    group = match.group(1)
                else:
                    group = 'Other'
                files_by_group[group].append(filename)
                
                # Create group directory
                group_dir = os.path.join(organized_dir, group)
                os.makedirs(group_dir, exist_ok=True)
                
                # Create symbolic link instead of copying (saves space)
                src = os.path.join(os.path.abspath(self.output_dir), filename)
                dst = os.path.join(os.path.abspath(group_dir), filename)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except:
                        # If symlink fails (e.g., on Windows), copy the file
                        import shutil
                        shutil.copy2(src, dst)
        
        # Print organization summary
        for group in sorted(files_by_group.keys()):
            files = files_by_group[group]
            print(f"\n{group}: {len(files)} documents")
            for filename in files[:3]:
                print(f"  - {filename}")
            if len(files) > 3:
                print(f"  ... and {len(files)-3} more")
        
        return files_by_group
    
    def create_catalog(self):
        """Create a catalog of all downloaded documents"""
        print(f"\n{'='*60}")
        print("Creating catalog...")
        print(f"{'='*60}")
        
        catalog = {
            "download_date": datetime.now().isoformat(),
            "total_documents": 0,
            "total_size_mb": 0,
            "documents": []
        }
        
        for filename in sorted(os.listdir(self.output_dir)):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.output_dir, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                
                # Extract ID from filename
                doc_id = None
                if 'oran_spec_' in filename:
                    try:
                        doc_id = int(filename.replace('oran_spec_', '').replace('.pdf', ''))
                    except:
                        pass
                
                catalog["documents"].append({
                    "filename": filename,
                    "size_mb": round(size_mb, 2),
                    "id": doc_id
                })
                catalog["total_size_mb"] += size_mb
        
        catalog["total_documents"] = len(catalog["documents"])
        catalog["total_size_mb"] = round(catalog["total_size_mb"], 2)
        
        # Save JSON catalog
        catalog_path = os.path.join(self.output_dir, "catalog.json")
        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
        
        # Save text list
        list_path = os.path.join(self.output_dir, "document_list.txt")
        with open(list_path, "w") as f:
            f.write(f"O-RAN Specifications Catalog\n")
            f.write(f"{'='*80}\n")
            f.write(f"Downloaded: {catalog['download_date']}\n")
            f.write(f"Total Documents: {catalog['total_documents']}\n")
            f.write(f"Total Size: {catalog['total_size_mb']} MB\n")
            f.write(f"{'='*80}\n\n")
            
            for doc in catalog["documents"]:
                f.write(f"{doc['filename']} ({doc['size_mb']} MB)\n")
        
        # Save successful IDs
        success_path = os.path.join(self.output_dir, "successful_ids.txt")
        with open(success_path, "w") as f:
            for doc_id, filename in sorted(self.successful):
                f.write(f"{doc_id}: {filename}\n")
        
        # Save failed IDs (non-404 only)
        failed_path = os.path.join(self.output_dir, "failed_ids.txt")
        with open(failed_path, "w") as f:
            for doc_id, error in sorted(self.failed):
                if "404" not in error:
                    f.write(f"{doc_id}: {error}\n")
        
        return catalog
    
    def print_summary(self, catalog):
        """Print final summary"""
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Successful downloads: {len(self.successful)}")
        print(f"✗ Failed attempts: {len(self.failed)}")
        print(f"  - 404 errors: {sum(1 for _, e in self.failed if '404' in e)}")
        print(f"  - Other errors: {sum(1 for _, e in self.failed if '404' not in e)}")
        print(f"\nTotal documents: {catalog['total_documents']}")
        print(f"Total size: {catalog['total_size_mb']} MB")
        print(f"\nFiles saved to: {os.path.abspath(self.output_dir)}")
        print(f"Catalog saved to: {os.path.abspath(os.path.join(self.output_dir, 'catalog.json'))}")
        print(f"Document list saved to: {os.path.abspath(os.path.join(self.output_dir, 'document_list.txt'))}")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Download O-RAN Specifications')
    parser.add_argument('--start', type=int, default=DEFAULT_START_ID, 
                        help=f'Start ID (default: {DEFAULT_START_ID})')
    parser.add_argument('--end', type=int, default=DEFAULT_END_ID, 
                        help=f'End ID (default: {DEFAULT_END_ID})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Parallel workers (default: {DEFAULT_WORKERS})')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between downloads in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--organize', action='store_true',
                        help='Organize files by Working Group')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: only download IDs 1-1000')
    
    args = parser.parse_args()
    
    # Quick mode overrides start/end
    if args.quick:
        args.start = 1
        args.end = 1000
    
    # Create downloader instance
    downloader = ORANDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        delay=args.delay
    )
    
    # Start downloading
    start_time = time.time()
    downloader.download_range(args.start, args.end)
    
    # Organize files if requested
    if args.organize:
        downloader.organize_files()
    
    # Create catalog
    catalog = downloader.create_catalog()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    downloader.print_summary(catalog)

if __name__ == "__main__":
    main()