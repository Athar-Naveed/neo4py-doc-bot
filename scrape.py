# import requests as req
# from bs4 import BeautifulSoup

# url = 'https://neo4pydocs.vercel.app/docs'

# response = req.get(url)

# if response.status_code == 200:
#     soup = BeautifulSoup(response.text, 'html.parser')
#     with open('neo4j_docs.html', 'w') as file:
#         file.write(soup.prettify())
#         print('File created successfully!')

# else:
#     print('Error:', response.status_code) 


# from firecrawl import FirecrawlApp
# from dotenv import load_dotenv
# load_dotenv()
# import os

# firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
# app = FirecrawlApp(api_key=firecrawl_api_key)

# response = app.scrape_url(url='https://neo4pydocs.vercel.app/docs', params={
# 	'formats': [ 'markdown' ],
# })

# with open('neo4pydocs.md', 'w') as file:
#     file.write(response['markdown'])

# print(response['markdown'])