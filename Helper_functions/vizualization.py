import pandas as pd

def triplet_table(triplets, xindex, yindex):
    """
    Erstellt eine Pivot-Tabelle aus Tripeln.
    
    Parameter:
        triplets (list of tuple): Liste von Tripeln mit numerischen Werten.
        xindex (int): Index (0, 1 oder 2), der für die x-Achse (Spalten) verwendet wird.
        yindex (int): Index (0, 1 oder 2), der für die y-Achse (Zeilen) verwendet wird.
    
    Rückgabe:
        pandas.DataFrame: Pivot-Tabelle mit den verbleibenden Werten.
    """
    assert xindex in [0, 1, 2]
    assert yindex in [0, 1, 2]
    assert xindex != yindex

    # Bestimme den Index der Wert-Spalte
    all_indices = {0, 1, 2}
    value_index = list(all_indices - {xindex, yindex})[0]

    # Konvertiere zur DataFrame
    df = pd.DataFrame(triplets, columns=["A", "B", "C"])
    
    # Benenne sinnvoll um
    col_names = [None, None, None]
    col_names[xindex] = "x"
    col_names[yindex] = "y"
    col_names[value_index] = "value"
    df.columns = col_names

    # Pivot-Tabelle erzeugen
    table = df.pivot(index="y", columns="x", values="value")
    return table

if __name__ == "__main__":
        # Beispielaufruf der Funktion
        # Triplets: (value, x, y)
    triplets = [
        (0.1, 1, 2),
        (0.2, 2, 2),
        (0.3, 1, 3),
        (0.4, 2, 3),
    ]

    table = triplet_table(triplets, 1, 2)
    print(table)
