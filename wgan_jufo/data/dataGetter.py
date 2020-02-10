import xlrd as x
import numpy as np

DATA_FILE = "augeralle.xlsx"


class dataGetter():

    # Variable für Datenquelle initialisieren
    def reCol(c):
        '''
        Methode zur Erfassung und Rückgabe der Ereignisdaten aus einer Exceltabelle
        '''
        wb = x.open_workbook(DATA_FILE)
        s = wb.sheet_by_index(0)
        col = np.asarray([s.cell(c, r).value for r in range(1, s.ncols)])
        return col

    def reNRows(file, nS):
        wb = x.open_workbook(file)
        s = wb.sheet_by_index(nS)
        return s.nrows


dG = dataGetter()
print(dG.reCol(2))
