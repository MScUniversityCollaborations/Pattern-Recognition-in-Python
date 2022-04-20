import pandas as pd


def read_from_csv():

    # Διαβάζουμε τα δεδομένα μας από τα csv αρχεία
    data = pd.read_csv('Data.csv', sep=',')
    matches_table = pd.read_csv("Matches.csv")

    # Δημιουργία των pandas dataframes για κάθε στοιχηματική
    b365 = matches_table[["win", "B365H", "B365D", "B365A"]]
    bw = matches_table[["win", "BWH", "BWD", "BWA"]]
    iw = matches_table[["win", "IWH", "IWD", "IWA"]]
    lb = matches_table[["win", "LBH", "LBD", "LBA"]]
    data = data[
            ["win", "buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
             "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth", "buildUpPlaySpeed",
             "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting",
             "defencePressure", "defenceAggression", "defenceTeamWidth", "B365H", "B365D", "B365A", "BWH", "BWD", "BWA",
             "IWH", "IWD", "IWA", "LBH", "LBD", "LBA"]]

    return b365, bw, iw, lb, data