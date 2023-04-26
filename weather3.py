from requests_html import HTMLSession



s = HTMLSession()



query = input('City\t')

City = print('The entered location is : ' +query)

url =f'https://www.google.com/search?q=weather+{query}'

r = s.get(url, headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'})

#temperature 
temp = r.html.find('span#wob_tm', first=True).text
#unit 
unit = r.html.find('div.vk_bk.wob-unit span.wob_t' , first=True).text
#description
desc = r.html.find('div.VQF4g' , first = True).find('span#wob_dc' , first = True).text
#humidity 
hum = r.html.find('div.wtsRwe' , first = True).find('span#wob_hm' , first = True).text

print(query)
print('temperature: '+temp,unit)
print('description: '+desc)
print('humidity: '+hum)
