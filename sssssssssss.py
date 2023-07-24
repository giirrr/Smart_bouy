from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import pandas as pd

# Create Chrome driver
driver = webdriver.Chrome()

url = "https://ticket.interpark.com/ConcertIndex.asp"
driver.get(url)
time.sleep(1)

elem = driver.find_element(By.ID,"Nav_SearchWord") 
elem.send_keys("콘서트") ## 원하는 내용
time.sleep(1)
#elem.send_keys(Keys.RETURN)
driver.find_element(By.CLASS_NAME,"btn_search").click()   ## 엔터키로 함
time.sleep(3)