import numpy as np
import sqlite3
import logging

class dataGrabber():
    '''
        Greift auf beliebige Datenbanken und Tabellen innerhalb dieser zu
        und instanziert einen Logger
    '''
    def __init__(self, DB_FILE):
        '''
            Initialisierung der Verbindung
        '''
        try:
            self.conn = sqlite3.connect(DB_FILE)
            self.c = self.conn.cursor()
            logging.basicConfig(filename='%Y%m%d-%H%M%S.log')
            logging.info('Verbindung zu Datenbank "' + DB_FILE + '" wurde hergestellt, Cursor steht zur Verfügung...')
        except sqlite3.ProgrammingError:
            logging.ERROR('Verbindung zur Datenbank "' + DB_FILE + '" nicht möglich...')



    def grabRow(self, TABLE_NAME, ROW_NUMBER):
        '''
            Gibt eine Reihe aus einer beliebigen Tabelle als NumPy Float-Array zurück
        '''
        ROW_ARRAY = np.array([], dtype='f')
        self.c.execute("SELECT * FROM {} WHERE EventID={}".format(TABLE_NAME, ROW_NUMBER))
        ROW_ARRAY = self.c.fetchone()
        print(ROW_ARRAY)
        return ROW_ARRAY


    def reNbRows(self, TABLE_NAME):
        '''
            Gibt die Anzhal an Reihen in einer Table zurück
        '''
        self.c.execute("SELECT * FROM {}".format(TABLE_NAME))
        RESULT_LIST = self.c.fetchall()
        NUMBER_OF_ROWS = len(RESULT_LIST)
        return NUMBER_OF_ROWS
    
    
    def reNbColumns(self, TABLE_NAME):
        '''
            Gibt die Anzahl an Spalten innerhalb einer Table wieder
        '''
        self.c.execute("PRAGMA table_info(%s)" % TABLE_NAME)
        NUMBER_OF_COLUMNS = len(self.c.fetchall())
        return NUMBER_OF_COLUMNS