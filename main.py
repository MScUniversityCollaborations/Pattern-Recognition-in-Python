import linearNN
import multilayerNN
import cmeans
import readData
from readData import read_from_csv

if __name__ == '__main__':

    # Προετοιμασία δεδομένων
    b365, bw, iw, lb, data = read_from_csv()

    print("Ερώτημα 1: ")
    # mean_loss_b365 = linearNN.input_data(b365)
    # mean_loss_bw = linearNN.input_data(bw)
    # mean_loss_iw = linearNN.input_data(iw)
    # mean_loss_lb = linearNN.input_data(lb)
    #
    # var = {mean_loss_b365: "B365",
    #        mean_loss_bw: "BW",
    #        mean_loss_iw: "IW",
    #        mean_loss_lb: "LB"}
    #
    # var_name = {"B365": mean_loss_b365,
    #             "BW": mean_loss_bw,
    #             "IW": mean_loss_iw,
    #             "LB": mean_loss_lb}
    #
    # best_company = var.get(min(var))
    # worst_company = var.get(max(var))
    #
    # print("Η καλύτερη  εταιρεία προβλέψεων είναι η: ", best_company, "με μέσο σφάλμα: ", var_name.get(best_company))
    # print("Η χειρότερη εταιρεία προβλέψεων είναι η: ", worst_company, "με μέσο σφάλμα: ", var_name.get(worst_company), "\n")

    print("Ερώτημα 2: ")
    # final_accuracy_b365, lowest_loss_b365 = multilayerNN.input_data(b365, 3)
    # final_accuracy_bw, lowest_loss_bw = multilayerNN.input_data(bw, 3)
    # final_accuracy_iw, lowest_loss_iw = multilayerNN.input_data(iw, 3)
    # final_accuracy_lb, lowest_loss_lb = multilayerNN.input_data(lb, 3)
    # var = {final_accuracy_b365: "B365",
    #        final_accuracy_bw: "BW",
    #        final_accuracy_iw: "IW",
    #        final_accuracy_lb: "LB"}
    #
    # var_name = {"B365": final_accuracy_b365,
    #             "BW": final_accuracy_bw,
    #             "IW": final_accuracy_iw,
    #             "LB": final_accuracy_lb}
    #
    # best_company = var.get(min(var))
    # worst_company = var.get(max(var))
    #
    # print("Η καλύτερη  εταιρεία προβλέψεων είναι η : ", best_company,  "με accuracy: ", var_name.get(best_company))
    # print("Η χειρότερη εταιρεία προβλέψεων είναι η : ", worst_company, "με accuracy: ", var_name.get(worst_company), "\n")

    print("Ερώτημα 3: ")
    # final_accuracy, lowest_loss = multilayerNN.input_data(data, 28)
    # print("Η συνάρτηση επιστέφει accuracy: ", final_accuracy, "και lowest loss: ", lowest_loss, "\n")

    print("Ερώτημα 4: Αναμονή για εμφάνιση γραφημάτων...")
    # b365.drop('win', 1, inplace=True)
    # cmeans.input_data(b365, 'B365')
    #
    # bw.drop('win', 1, inplace=True)
    # cmeans.input_data(bw, "BW")
    #
    # iw.drop('win', 1, inplace=True)
    # cmeans.input_data(iw, "IW")

    #lb.drop('win', 1, inplace=True)
    #cmeans.input_data(lb, "LB")
