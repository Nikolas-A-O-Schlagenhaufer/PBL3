import cv2
import numpy as np
from math import gcd
import statistics
from alive_progress import alive_bar

def create_masks(img, altura:int=9, largura:int=16):
	mdc = gcd(img.shape[0], img.shape[1])
	print(mdc)
	masks = {}
	for linha in range(altura):
		for coluna in range(largura):
			masks[(linha,coluna)] = np.zeros(img.shape[:2], np.uint8)
			masks[(linha,coluna)][linha*mdc:(linha+1)*mdc, coluna*mdc:(coluna+1)*mdc] = 255
	return masks

def create_hists(img, masks, altura:int=9, largura:int=16):
	masked_imgs = {}
	histograms = {}
	for linha in range(altura):
		for coluna in range(largura):
			masked_imgs[(linha,coluna)] = cv2.bitwise_and(img,img,mask = masks[(linha,coluna)])
			histograms[(linha,coluna)] = cv2.calcHist([img],[0],masks[(linha,coluna)],[256],[0,256])
	return histograms, masked_imgs

def calc_hist_mean(histograms, altura:int=9, largura:int=16):
	size = len(histograms[0,0])
	hist_mean = {}
	for linha in range(altura):
		for coluna in range(largura):
			sum_result = [histograms[(linha,coluna)][i][0] for i in range(size)]
			hist_mean[(linha,coluna)] = sum_result.index(max(sum_result))
	return hist_mean

def pixel_mean(img, altura:int=9, largura:int=16):
	mdc = gcd(img.shape[0], img.shape[1])
	means = {}
	for linha in range(altura):
		for coluna in range(largura):
			means[(linha,coluna)] = int(statistics.mean(np.array(img[linha*mdc:(linha+1)*mdc, coluna*mdc:(coluna+1)*mdc].flatten())))
	return means

def verify_pixel_means(pixel_means, altura:int=9, largura:int=16):
	results = {}
	for linha in range(altura):
		for coluna in range(largura):
			if pixel_means[(linha,coluna)] >= 150:
				results[(linha,coluna)] = 0
			else:
				results[(linha,coluna)] = 1
	return results

def create_result(img, results, altura:int=9, largura:int=16):
	mdc = gcd(img.shape[0], img.shape[1])
	result = np.zeros(img.shape[:2], np.uint8)
	for linha in range(altura):
		for coluna in range(largura):
			result[linha*mdc:(linha+1)*mdc, coluna*mdc:(coluna+1)*mdc] = results[(linha,coluna)]*255
	return result

def main():
	with alive_bar(29) as bar:
		for nr in range(29):
			img = cv2.imread(f'yolov5-master/frame{nr}.png', cv2.IMREAD_GRAYSCALE)
			pixel_means = pixel_mean(img)
			results = verify_pixel_means(pixel_means)
			frame = create_result(img, results)
			cv2.imwrite(f"found-frames/frame{nr}.png", frame)
			bar()



if __name__ == "__main__":
	main()