# Author: Jacob Dawson
#
# The idea for this script is to just scrape images available to the public on the Junocam website.
# Each image is available at a link like the following:
# https://www.missionjuno.swri.edu/junocam/processing?id=13190
# The above image is a user-made image.
# Importantly, we want this script to be able to put user-made images in one folder, and raw data in another.
# This is because the GAN will need to have the two categories separate for training.
# Note: this script breaks easily, and I had to mess with it a lot to get a proper dataset to work with.
# Good luck!

from bs4 import BeautifulSoup # this will be used to navigate the website
import requests # grabs the HTML
#import os # saves the images
from PIL import Image # saves the images

no_user_images = True # this is a parameter you can set to ignore all user images.
# I've set it to true so that we're only downloading RAW data.

# we can alter the size/qualities of our generated dataset by just declaring the for-loop differently.
#for i in range(13190, 0, -1): # this iterates through the whole thing. It'll probably create a huge dataset. Not recommended!
for i in range(13190, 0, -1): # this will save every 50th image.
	# i is the pagenumber we're looking at.
	URL = "https://www.missionjuno.swri.edu/junocam/processing?id="
	URL += str(i)
	print(URL, end='\n')
	page = requests.get(URL)
	soup = BeautifulSoup(page.content, "html.parser")
	im_div = soup.find_all('div',class_='processing_tools half stack_full textright padT_half right')
	im_exists = (1==len(im_div))
	if not im_exists:
		print(i,"was deleted.")
		print("Finished with", i)
		continue # there are deleted images. Skip those.
	im_div = im_div[0] # take it out of list form.
	#print(im_div, end='\n')

	user_or_nasa = (2==len(soup.find_all('button', class_='dropdown_toggle')))
	# user_or_nasa will be false if user-uploaded, true if nasa-uploaded
	print("This page has the raw data:",user_or_nasa, end='\n')

	im_page = 'http://missionjuno.swri.edu'
	link = im_div.find('a', class_="marR")
	#print(link)
	# we have isolated the link. Now we just need to do some python string manipulation to complete the url.
	
	#print(type(link))
	#print(type(link.get('href')))
	#print(link.get('href'))
	im_page+=link.get('href')
	print(im_page)
	# success! im_page is now the url of the image we want to download.

	#img = Image.open(requests.get(im_page, stream=True).raw)

	if user_or_nasa:
		# in this case, we're dealing with a raw image
		img = Image.open(requests.get(im_page, stream=True).raw)
		img.save('raw_imgs/'+str(i)+'.png')
	else:
		if no_user_images:
			# change this on line 13
			print("Finished with", i)
			continue

		# in this case, we're dealing with something a human uploaded
		img = Image.open(requests.get(im_page, stream=True).raw)
		img.save('user_imgs/'+str(i)+'.png')
	print("Finished with", i)
	#print('\n\n')