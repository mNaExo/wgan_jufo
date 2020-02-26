import xlrd as x
import numpy as np

# DATA_FILE = "augeralle.xlsx"


class dataGetter():
    '''
    Her mit den Daten, sonst komme ich mit der Peitsche!
    '''

    def __init__(self, file):
        self.DATA_FILE = file
        print("Lets get this Dataparty started on the dancefloor. Datei, aus der gezogen wird: " + file)

    # Variable für Datenquelle initialisieren
    def reCol(self, c):
        '''
        Methode zur Erfassung und Rückgabe der Ereignisdaten aus einer Exceltabelle
        '''
        wb = x.open_workbook(self.DATA_FILE)
        s = wb.sheet_by_index(0)
        col = np.asarray([s.cell(c, r).value for r in range(1, s.ncols)])
        return col

    def reNRows(self, nS):
        wb = x.open_workbook(self.DATA_FILE)
        s = wb.sheet_by_index(nS)
        return s.nrows
