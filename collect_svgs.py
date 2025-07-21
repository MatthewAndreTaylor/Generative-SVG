import re
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm

session = None


def get_page(driver):
    try:
        page_footer = driver.find_element(By.CSS_SELECTOR, 'div[class^="style_pagingCarrier"]').text
        return int(re.sub(r'.*/\s+', r'', page_footer))
    except Exception:
        return 1


def download_items(all_links, download_path, bar):
    for link in all_links:
        aid = os.path.basename(os.path.dirname(link))
        dest = os.path.join(download_path, aid + '-' + os.path.basename(link))
        if os.path.exists(dest):
            continue
        x = session.get(link)
        if x.headers.get('content-type') != 'image/svg+xml':
            print(f"err: {link}")
            continue
        with open(dest, 'wb') as f:
            f.write(x.content)
        bar.update(1)


def downloader(downlod_path, url='https://www.svgrepo.com/collection/chunk-16px-thick-interface-icons/'):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    os.makedirs(downlod_path, exist_ok=True)
    num_page = get_page(driver)
    
    global session
    session = requests.Session()

    for page in range(1, num_page + 1):
        if page > 1:
            driver.get(url + str(page))
            time.sleep(1)

        imgs = driver.find_elements(By.CSS_SELECTOR, 'div[class^="style_NodeImage_"] img[itemprop="contentUrl"]')
        all_links = [img.get_attribute('src') for img in imgs if img.get_attribute('src')]
        
        if not all_links:
            break

        bar = tqdm(total=len(all_links), desc=f"Download Page {page}")
        download_items(all_links, downlod_path, bar)
        bar.close()
    
    driver.quit()
    
    
# url = f'https://www.svgrepo.com/collections/{category}/'

def list_collections(url='https://www.svgrepo.com/collections/all/'):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    num_page = get_page(driver)
    for page in range(1, num_page + 1):
        print(f'Listing Collections {page}/{num_page}')
        if page > 1:
            driver.get(url + str(page))
            time.sleep(1)

        links = driver.find_elements(By.CSS_SELECTOR, 'div[class^="style_Collection__"] a')
        for link in links:
            yield link.get_attribute('href')
    driver.quit()
    

def minify_svg(file_path):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(), options=options)

    try:
        # Go to the SVG minifier page and wait for it to load
        driver.get("https://devina.io/svg-minifier")
        time.sleep(1)

        # Locate the file input (most likely hidden, so we use JS or locate it directly)
        file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")

        # Upload the file and wait for processing
        file_input.send_keys(file_path)
        time.sleep(1)

        svg_iframe = driver.find_element(By.CSS_SELECTOR, 'iframe[title="SVG"]')
        driver.switch_to.frame(svg_iframe)
        svg_element = driver.find_element(By.CSS_SELECTOR, "svg")
        svg_content = svg_element.get_attribute("outerHTML")
        
        # find the difference in file size
        original_size = os.path.getsize(file_path)
        minified_size = len(svg_content.encode('utf-8'))
        print(f"Original size: {original_size} bytes, Minified size: {minified_size} bytes")
        
        os.remove(file_path)  # Remove the original file
        new_file_path = os.path.join(os.path.dirname(file_path), f"minified_{os.path.basename(file_path)}")

        # Save to file
        with open(new_file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

    finally:
        driver.quit()


if __name__ == "__main__":
    
    for collection in list_collections('https://www.svgrepo.com/collections/all/'):
        slug = collection.rstrip('/').split('/')[-1]
        
        # Note: this will skip the collection if it already exists
        if os.path.exists(f'data/svgs/{slug}'):
            print(f"Collection {slug} already exists, skipping download.")
            continue
        
        print(f"Processing collection: {collection}, slug: {slug}")
        downloader(f'data/svgs/{slug}', collection)
        
        for svg_file in os.listdir(f'data/svgs/{slug}'):
            if svg_file.endswith('.svg'):
                file_path = os.path.join(f'data/svgs/{slug}', svg_file)
                absolute_path = os.path.abspath(file_path)
                
                # Check if minified_{svg_file} already exists
                if os.path.exists(os.path.join(f'data/svgs/{slug}', f'minified_{svg_file}')):
                    print(f"Minified version already exists for {svg_file}, skipping minification.")
                    continue
                
                print(f"Minifying SVG: {absolute_path}")
                minify_svg(absolute_path)