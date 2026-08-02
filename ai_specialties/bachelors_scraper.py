import csv
import random
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


# Wait and close the cookies dialog
def close_cookies_dialog(driver, wait):
    try:
        cookie_dialog = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'cookiefirst-root')))
        adjust_cookies_btn = driver.find_element(By.CSS_SELECTOR, 'div.cf1lHZ:nth-child(2) > button:nth-child(1)')
        time.sleep(1)
        adjust_cookies_btn.click()
    except Exception as e:
        print('Cookies did not close properly')

def create_driver():
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(120)
    return driver

# Collect data
data = []

# Base url
base_url = 'https://www.bachelorsportal.com/search/bachelor?redirect=false&kw=Artificial+Intelligence'

# Browse pages
# 1 - 29 - DONE 911
# 29 - 54 - DONE 1411
# 54 - 82 - DONE 1971
# 82 - 114 - DONE 2598
cur_page = 82
last_page = 114

start_time = time.time()

for page in range(cur_page, last_page):
    print(f'Scraping page {page}')

    rand_wait_in_sec = random.randint(1, 8)
    time.sleep(rand_wait_in_sec)

    driver = create_driver()
    wait = WebDriverWait(driver, 120)

    if page == 1:
        driver.get(base_url)
    else:
        page_url = base_url + '&page=' + str(page)
        driver.get(page_url)

    # Close cookies (if displayed)
    close_cookies_dialog(driver, wait)

    # Scrape the programs list
    program_cards = driver.find_elements(By.CLASS_NAME, 'ProgrammeCard')
    print(f'Found {len(program_cards)} programs')

    for card in program_cards:
        program_name = card.find_element(By.CSS_SELECTOR, 'h2')
        university_name = card.find_element(By.CLASS_NAME, 'OrganizationName')
        location_name = card.find_element(By.CLASS_NAME, 'Locations')
        record = [program_name.text, university_name.text, location_name.text, 'B']
        print(record)
        data.append(record)

    driver.quit()

end_time = time.time()

print(f'Total time: {end_time - start_time} seconds')

# Save data to csv
with open('output.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
