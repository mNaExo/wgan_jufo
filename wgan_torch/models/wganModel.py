import data.dataGrabber
import numpy as np

class wganModel():
    '''
        Hier kommt noch was Schlaues hin...
    '''
    def __init__(self, DB_FILE, DB_NAME):
        self.dataGrabber = data.dataGrabber(DB_FILE, DB_NAME)
        ALL_EVENTS = np.array([])