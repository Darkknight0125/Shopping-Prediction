import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def give_month(month):
    if month == "Jan": return 0
    elif month == "Feb" : return 1
    elif month == "Mar" : return 2
    elif month == "Apr" : return 3
    elif month == "May" : return 4
    elif month == "Jun" : return 5
    elif month == "Jul" : return 6
    elif month == "Aug" : return 7
    elif month == "Sep" : return 8
    elif month == "Oct" : return 9
    elif month == "Nov" : return 10
    elif month == "Dec" : return 11
    else: return 12

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        0- Administrative, an integer
        1- Administrative_Duration, a floating point number
        2- Informational, an integer
        3- Informational_Duration, a floating point number
        4- ProductRelated, an integer
        5- ProductRelated_Duration, a floating point number
        6- BounceRates, a floating point number
        7- ExitRates, a floating point number
        8- PageValues, a floating point number
        9- SpecialDay, a floating point number
        10- Month, an index from 0 (January) to 11 (December)
        11- OperatingSystems, an integer
        12- Browser, an integer
        13- Region, an integer
        14- TrafficType, an integer
        15- VisitorType, an integer 0 (not returning) or 1 (returning)
        16- Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = list()
    lables = list()
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            e = list()
            e.append(int(row[0]))
            e.append(float(row[1]))
            e.append(int(row[2]))
            e.append(float(row[3]))
            e.append(int(row[4]))
            for i in [5, 6, 7, 8, 9]:
                e.append(float(row[i]))
            e.append(int(give_month(row[10])))
            for i in [11, 12, 13, 14]:
                e.append(int(row[i]))
            if row[15] == "Returning_Visitor":
                e.append(1)
            else: e.append(0)
            if row[16] == "FALSE":
                e.append(0)
            else: e.append(1)
            evidence.append(e)

            if row[17] == "TRUE":
                lables.append(1)
            else: lables.append(0)

    return(evidence, lables)
    #raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(evidence, labels)
    return model
    #raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    total = len(labels)
    for i in range(total):
        if labels[i] == 1:
            if predictions[i] == 1:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if predictions[i] == 0:
                true_neg += 1
            else:
                false_pos += 1
    
    sensitivity = float(true_pos) / float(true_pos + false_neg)
    specificity = float(true_neg) / float(true_neg + false_pos)
    return (sensitivity, specificity)
    #raise NotImplementedError


if __name__ == "__main__":
    main()
