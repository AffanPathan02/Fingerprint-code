import fingerprint_feature_extractor
import cv2
img = cv2.imread('Sample-Fingerprint-Image.png', 0)				# read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)