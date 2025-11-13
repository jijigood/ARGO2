import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(doc_id, session, output_dir="oran_specs"):
    """Download a single file by ID"""
    url = f"https://specifications.o-ran.org/download?id={doc_id}"
    
    try:
        response = session.get(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            # Try to get filename from Content-Disposition header
            filename = None
            if 'content-disposition' in response.headers:
                cd = response.headers['content-disposition']
                if 'filename=' in cd:
                    filename = cd.split('filename=')[-1].strip('"')
            
            if not filename:
                filename = f"oran_spec_{doc_id}.pdf"
            
            filepath = os.path.join(output_dir, filename)
            
            # Download the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return doc_id, True, filename
        else:
            return doc_id, False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return doc_id, False, str(e)

def main():
    # Create output directory
    os.makedirs("oran_specs", exist_ok=True)
    
    # Define the range of IDs to try
    # You can modify this based on what you discover
    id_start = 1
    id_end = 1000
    
    # Or use a specific list of IDs
    # ids = [775, 776, 777, ...]  # Add your IDs here
    
    ids = list(range(id_start, id_end + 1))
    
    # Create a session for connection pooling
    session = requests.Session()
    
    successful = []
    failed = []
    
    # Use ThreadPoolExecutor for parallel downloads
    # Limit workers to be respectful to the server
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(download_file, doc_id, session): doc_id 
            for doc_id in ids
        }
        
        # Progress bar
        for future in tqdm(as_completed(futures), total=len(ids), desc="Downloading"):
            doc_id, success, result = future.result()
            
            if success:
                successful.append((doc_id, result))
                print(f"\n✓ Downloaded: {result}")
            else:
                failed.append((doc_id, result))
            
            # Small delay between completions
            time.sleep(0.5)
    
    # Summary
    print(f"\n\nDownload Summary:")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Save successful IDs for reference
    with open("successful_ids.txt", "w") as f:
        for doc_id, filename in successful:
            f.write(f"{doc_id}: {filename}\n")
    
    if failed:
        print("\nFailed IDs (might not exist):")
        with open("failed_ids.txt", "w") as f:
            for doc_id, error in failed[:10]:  # Show first 10
                print(f"  ID {doc_id}: {error}")
                f.write(f"{doc_id}: {error}\n")

if __name__ == "__main__":
    main()