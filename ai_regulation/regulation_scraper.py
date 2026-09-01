from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a separate file for each section
def create_file(section_name, content):
    file_path = f"./output/{section_name}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

# Initialize driver
driver = webdriver.Chrome()
driver.set_page_load_timeout(10)

# Initialize wait
wait = WebDriverWait(driver, 20)

# Set the url
base_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
driver.get(base_url)

# Get contents with all the links
table_of_contents = wait.until(EC.visibility_of_element_located((By.ID, "TOC")))
links = table_of_contents.find_elements(By.TAG_NAME, "a")

# Process each link and create a file with the text
for idx, link in enumerate(links):
    cleaned_href = link.get_attribute('href').replace(base_url, '').replace('#', '')
    print(f'Processing {cleaned_href}')
    article_text = driver.find_element(By.ID, cleaned_href).text
    create_file(section_name=cleaned_href, content=article_text)
    print(f'Successfully processed {cleaned_href}')

# Quit the driver
driver.quit()