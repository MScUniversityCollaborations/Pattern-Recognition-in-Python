import pandas as pd
import linearNN


def readCSV():
    # Reading data from csv file
    all_data = pd.read_csv('all_data.csv', sep=',')
    matches_table = pd.read_csv("Matches.csv")

    # Creating pandas dataframes
    b365 = matches_table[["win", "B365H", "B365D", "B365A"]]
    bw = matches_table[["win", "BWH", "BWD", "BWA"]]
    iw = matches_table[["win", "IWH", "IWD", "IWA"]]
    lb = matches_table[["win", "LBH", "LBD", "LBA"]]
    all_data = all_data[
        ["win", "buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing",
         "chanceCreationShooting", "defencePressure", "defenceAggression", "defenceTeamWidth", "buildUpPlaySpeed",
         "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting",
         "defencePressure", "defenceAggression", "defenceTeamWidth", "B365H", "B365D", "B365A", "BWH", "BWD", "BWA",
         "IWH", "IWD", "IWA", "LBH", "LBD", "LBA"]]

    return b365, bw, iw, lb, all_data


if __name__ == '__main__':
    linearNN.start()
    # b365, bw, iw, lb, all_data = readCSV()
