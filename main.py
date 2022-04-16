import pandas as pd

import linearNN
import linearNN2
import multilayerNN


def readCSV():
    # Reading data from csv file
    data = pd.read_csv('Data.csv', sep=',')
    matches_table = pd.read_csv("Matches.csv")

    # Creating pandas dataframes
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


if __name__ == '__main__':
    b365, bw, iw, lb, data = readCSV()
    print(b365)
    print(len(bw))
    # linearNN.start(b365)

    print("Ερώτημα 2: ")
    final_accuracy_b365, lowest_loss_b365 = multilayerNN.input_data(b365, 3)
    final_accuracy_bw, lowest_loss_bw = multilayerNN.input_data(bw, 3)
    final_accuracy_iw, lowest_loss_iw = multilayerNN.input_data(iw, 3)
    final_accuracy_lb, lowest_loss_lb = multilayerNN.input_data(lb, 3)
    var = {final_accuracy_b365: "B365",
           final_accuracy_bw: "BW",
           final_accuracy_iw: "IW",
           final_accuracy_lb: "LB"}

    best_company = var.get(min(var))
    worst_company = var.get(max(var))

    print("Η καλύτερη εταιρεία προβλέψεων είναι η : ", best_company, "\n")
    print("Η χειρότερη εταιρεία προβλέψεων είναι η : ", worst_company, "\n")

    print("Ερώτημα 3: ")
    final_accuracy, lowest_loss = multilayerNN.input_data(data, 28)
    print(final_accuracy, lowest_loss)