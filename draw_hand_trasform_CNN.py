import cv2
import numpy as np
from PIL import Image


class DrawHands(object):
	def __call__(self, pic):
		img = np.array(pic)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kernel = (5, 5)

		mask = img > 70
		img[mask] = 255

		closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

		dilated = cv2.dilate(closing, kernel, iterations=10)

		return Image.fromarray(dilated.astype(np.uint8))

	def __repr__(self):
		return self.__class__.__name__ + '()'
