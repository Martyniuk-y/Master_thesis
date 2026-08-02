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
base_url = 'https://www.mastersportal.com/search/master?kw-what=Artificial%20Intelligence'

# Browse pages
# 1 - 32 - DONE 3218
# 32 - 64 - DONE 3858
# 64 - 82 - DONE 4218
# 82 - 102 - DONE 4618
# 102 - 125 - DONE 5078
# 125 - 167 - DONE 5918
cur_page = 125
last_page = 167

start_time = time.time()

for page in range(cur_page, last_page):
    print(f'Scraping page {page}')

    rand_wait_in_sec = random.randint(2, 8)
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
    program_names = driver.find_elements(By.CLASS_NAME, 'StudyName')
    university_names = driver.find_elements(By.CLASS_NAME, 'OrganisationName')
    locations = driver.find_elements(By.CLASS_NAME, 'OrganisationLocation')
    print(f'Found {len(program_names)} programs')

    for elem in range(len(program_names)):
        print([program_names[elem].text, university_names[elem].text, locations[elem].text, 'M'])
        data.append([program_names[elem].text, university_names[elem].text, locations[elem].text, 'M'])
    driver.quit()

end_time = time.time()

print(f'Total time: {end_time - start_time} seconds')

# Save data to csv
with open('output.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
