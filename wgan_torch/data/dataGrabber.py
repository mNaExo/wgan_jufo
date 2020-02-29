import numpy as np
import sqlite3


class dataGrabber():
    '''
        Greift auf beliebige Datenbanken und Tabellen innerhalb dieser zu
    '''
    def __init__(self, DB_FILE):
        '''
            Initialisierung der Verbindung
        '''
        self.conn = sqlite3.connect(DB_FILE)
        self.c = self.conn.cursor()


    def grabRow(self, TABLE_NAME, ROW_NUMBER):
        '''
            Gibt eine Reihe aus einer beliebigen Tabelle als NumPy Float-Array zurück
        '''
        ROW_ARRAY = np.array([], dtype='f')
        self.c.execute("SELECT * FROM {} WHERE EventID={}".format(TABLE_NAME, ROW_NUMBER))
        ROW_ARRAY = self.c.fetchone()
        print(ROW_ARRAY)
