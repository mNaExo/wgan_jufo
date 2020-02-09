import xlrd as x
import numpy as np

# Einlesen der Excel-Datei, Rückgabe eines Ereignisses (einer Reihe) als Array

DATA_FILE = "augeralle.xlsx"
def reCol(c):
    wb = x.open_workbook(DATA_FILE)
    s = wb.sheet_by_index(0)
    col = np.asarray([s.cell(c, r).value for r in range(1, s.ncols])
    return col