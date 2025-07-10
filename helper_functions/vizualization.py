import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def plot_convergence(error_data, error_index, h_index, label="L2 error", title="Convergence Plot"):
    """
    Plot log-log convergence graph from arbitrary tuples by specifying indices.

    Parameters:
    - error_data: List of tuples containing convergence data
    - error_index: Index in the tuple where the L2 error is stored
    - h_index: Index in the tuple where the maxh (mesh size) is stored
    - label: Label for the plot line
    - title: Title of the plot
    """
    # Extract mesh sizes and errors
    maxhs = np.array([t[h_index] for t in error_data])
    errors = np.array([t[error_index] for t in error_data])

    # Sort by mesh size
    sorted_indices = np.argsort(maxhs)
    maxhs = maxhs[sorted_indices]
    errors = errors[sorted_indices]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.loglog(maxhs, errors, 'o-', label=label)

    # Compute and plot experimental order of convergence (EOC)
    if len(maxhs) >= 2:
        rates = np.log(errors[1:] / errors[:-1]) / np.log(maxhs[1:] / maxhs[:-1])
        avg_rate = np.mean(rates)
        plt.text(maxhs[1], errors[1], f"EOC ≈ {avg_rate:.2f}", fontsize=12)

    plt.xlabel("maxh")
    plt.ylabel("L2 error")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()



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
    # Beispielaufruf der Funktion plot_convergence
    errors_u = [(0.5, 0.12), (0.25, 0.03), (0.125, 0.0075)]
    plot_convergence(errors_u,error_index=1, h_index=0, label="L2 error of velocity")