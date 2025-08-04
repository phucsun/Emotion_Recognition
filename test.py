import cv2
import numpy as np
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt