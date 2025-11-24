#!/usr/bin/env python3
"""
O-RAN Specifications Bulk Downloader
Downloads all O-RAN specification documents from specifications.o-ran.org
"""

import argparse
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import unquote

import requests

# Configuration
DEFAULT_START_ID = 1
DEFAULT_END_ID = 1200
DEFAULT_WORKERS = 3
DEFAULT_OUTPUT_DIR = "ORAN_Docs"
DEFAULT_DELAY = 1.0
DEFAULT_BATCH_SIZE = 25
DEFAULT_STOP_AFTER_404 = 75

FILENAME_SANITIZE_PATTERN = re.compile(r'[<>:"/\\|?*\x00]')
METADATA_FILES = {"catalog.json", "document_list.txt", "successful_ids.txt", "failed_ids.txt"}

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
        
    def _sanitize_filename(self, filename, doc_id):
        if not filename:
            return f"oran_spec_{doc_id}.bin"
        cleaned = FILENAME_SANITIZE_PATTERN.sub('_', filename.strip())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or f"oran_spec_{doc_id}.bin"

    def _extract_filename(self, headers, doc_id):
        content_disposition = headers.get('content-disposition', '')
        filename = None

        if content_disposition:
            star_match = re.search(r"filename\*=(?:UTF-8''|)([^;]+)", content_disposition, flags=re.I)
            if star_match:
                candidate = star_match.group(1).strip().strip('"')
                filename = unquote(candidate)
            else:
                regular_match = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.I)
                if regular_match:
                    filename = regular_match.group(1).strip()

        if not filename:
            content_type = headers.get('content-type', 'application/octet-stream').split(';')[0].strip()
            extension = mimetypes.guess_extension(content_type) or '.bin'
            filename = f"oran_spec_{doc_id}{extension}"

        return self._sanitize_filename(filename, doc_id)

    def _download_ids(self, ids, label=None):
        if not ids:
            return []

        total = len(ids)
        label_prefix = f"{label} " if label else ""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_file, doc_id): doc_id for doc_id in ids}
            completed = 0

            for future in as_completed(futures):
                completed += 1
                doc_id, success, result = future.result()

                if success:
                    self.successful.append((doc_id, result))
                else:
                    self.failed.append((doc_id, result))

                progress = completed / total * 100
                status = '✓' if success else '✗'
                if success or ("404" not in result):
                    print(f"{label_prefix}[{completed}/{total} - {progress:.1f}%] {status} ID {doc_id}: {result}")

                results.append((doc_id, success, result))
                time.sleep(self.delay)

        results.sort(key=lambda item: item[0])
        return results
        
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
            
            if response.status_code == 200 and 'content-disposition' in response.headers:
                filename = self._extract_filename(response.headers, doc_id)
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
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
        self._download_ids(ids)
        return self.successful, self.failed

    def auto_download(self, start_id, batch_size=DEFAULT_BATCH_SIZE, stop_after=DEFAULT_STOP_AFTER_404, max_id=None):
        """Download sequential IDs until a large block of 404s is observed."""
        print(f"\n{'='*60}")
        print("O-RAN Specifications Downloader (auto mode)")
        print(f"{'='*60}")
        print(f"Starting at ID {start_id}")
        if max_id is not None:
            print(f"Hard stop at ID {max_id}")
        print(f"Batch size: {batch_size}")
        print(f"Stop after {stop_after} consecutive 404s")
        print(f"Output directory: {self.output_dir}")
        print(f"Parallel workers: {self.max_workers}")
        print(f"{'='*60}")

        current = start_id
        consecutive_404 = 0
        batch_index = 1

        while True:
            if max_id is not None and current > max_id:
                break

            batch_end = current + batch_size - 1
            if max_id is not None:
                batch_end = min(batch_end, max_id)

            ids = list(range(current, batch_end + 1))
            if not ids:
                break

            print(f"\n--- Batch {batch_index}: IDs {current}-{batch_end} ---")
            results = self._download_ids(ids, label=f"Batch {batch_index}")

            for _, success, result in results:
                if success:
                    consecutive_404 = 0
                elif "404" in result:
                    consecutive_404 += 1
                else:
                    consecutive_404 = 0

            if (max_id is not None and batch_end >= max_id) or consecutive_404 >= stop_after:
                if consecutive_404 >= stop_after:
                    print(f"Reached {stop_after} consecutive HTTP 404 responses. Assuming catalog end.")
                break

            current = batch_end + 1
            batch_index += 1

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
        
        valid_extensions = ('.pdf', '.doc', '.docx', '.zip', '.xlsx', '.xls', '.pptx')

        for filename in os.listdir(self.output_dir):
            if filename in METADATA_FILES:
                continue

            filepath = os.path.join(self.output_dir, filename)
            if not os.path.isfile(filepath):
                continue

            if not filename.lower().endswith(valid_extensions):
                continue

            match = pattern.search(filename)
            group = match.group(1) if match else 'Other'
            files_by_group[group].append(filename)

            group_dir = os.path.join(organized_dir, group)
            os.makedirs(group_dir, exist_ok=True)

            src = os.path.abspath(filepath)
            dst = os.path.join(os.path.abspath(group_dir), filename)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except:
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
        id_lookup = {filename: doc_id for doc_id, filename in self.successful}
        
        for filename in sorted(os.listdir(self.output_dir)):
            if filename in METADATA_FILES:
                continue

            filepath = os.path.join(self.output_dir, filename)
            if not os.path.isfile(filepath):
                continue

            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            catalog["documents"].append({
                "filename": filename,
                "size_mb": round(size_mb, 2),
                "id": id_lookup.get(filename)
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
                prefix = f"ID {doc['id']}: " if doc['id'] is not None else ""
                f.write(f"{prefix}{doc['filename']} ({doc['size_mb']} MB)\n")
        
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
    parser.add_argument('--auto', action='store_true',
                        help='Automatically continue downloading sequential IDs until stop criteria are met')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for --auto mode (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--stop-after', type=int, default=DEFAULT_STOP_AFTER_404,
                        help=f'Number of consecutive 404 responses before stopping auto mode (default: {DEFAULT_STOP_AFTER_404})')
    parser.add_argument('--max-id', type=int, default=None,
                        help='Hard upper bound when using --auto (inclusive). Default: no limit')
    
    args = parser.parse_args()
    
    # Quick mode overrides start/end
    if args.quick:
        args.start = 1
        args.end = 1000
        if args.auto and args.max_id is None:
            args.max_id = 1000
    
    # Create downloader instance
    downloader = ORANDownloader(
        output_dir=args.output,
        max_workers=args.workers,
        delay=args.delay
    )
    
    # Start downloading
    start_time = time.time()
    if args.auto:
        downloader.auto_download(
            start_id=args.start,
            batch_size=args.batch_size,
            stop_after=args.stop_after,
            max_id=args.max_id
        )
    else:
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