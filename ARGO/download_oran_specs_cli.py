#!/usr/bin/env python3
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://www.o-ran.org/specifications"
SAVE_DIR = "o-ran-official-specs"
os.makedirs(SAVE_DIR, exist_ok=True)

print("正在访问 O-RAN 官方规范页面...")
response = requests.get(BASE_URL)
if response.status_code != 200:
    print(f"错误：无法访问页面，状态码 {response.status_code}")
    exit(1)

soup = BeautifulSoup(response.text, 'html.parser')

download_urls = set()
pattern = re.compile(r'\.(docx|pdf)$', re.I)

# 提取 <a href="..."> 中的链接
for a in soup.find_all('a', href=True):
    href = a['href']
    if pattern.search(href):
        full_url = urljoin(BASE_URL, href)
        if "download" in full_url.lower() or "specifications" in full_url:
            download_urls.add(full_url)

# 提取 JavaScript 中的真实下载链接（关键！）
scripts = soup.find_all('script')
for script in scripts:
    if script.string:
        matches = re.findall(r"window\.location\s*=\s*['\"]([^'\"]+?\.(docx|pdf))['\"]", script.string, re.I)
        for url, _ in matches:
            full_url = urljoin(BASE_URL, url)
            download_urls.add(full_url)

print(f"共发现 {len(download_urls)} 个文档链接")

# 下载
success = 0
for url in sorted(download_urls):
    filename = os.path.basename(url.split('?')[0])
    if not pattern.search(filename):
        continue
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        print(f"跳过已存在: {filename}")
        success += 1
        continue

    print(f"下载中: {filename}")
    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"完成: {filename} ({size_mb:.1f} MB)")
            success += 1
        else:
            print(f"失败: {filename} (HTTP {r.status_code})")
    except Exception as e:
        print(f"错误: {filename} -> {e}")
    time.sleep(0.5)

print(f"\n下载完成！共 {success}/{len(download_urls)} 个文件保存在: {SAVE_DIR}/")
