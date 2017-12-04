#This file contains the main decision tree functions

# Defining our model : 

# Column labels : 
header = ["WaterQuantity", "area", "Region", "Gain", "label"]

""" This dataset will be read from an csv file using panda """
training_data = [
    ['lot',1000,'Setif',  100, 'Apple'],
    ['little',1000,'Setif', 30, 'Apple'],
    ['lot', 1000,'Setif', 100, 'Grape'],
    ['lot', 1000,'Setif', 10, 'Grape'],
    ['little', 1000,'Setif', 39, 'Lemon'],
]

def uniqueValues(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # Associate label -> count 
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def isNum(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset."""

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if isNum(val):
            return val >= self.value
        elif val == "lot" : 
            return 1 
        elif val == "little" : 
            return 0 
        # elif (Region) call an API.  


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    rows1, rows0 = [], []
    for row in rows:
        if question.match(row):
            rows1.append(row)
        else:
            rows0.append(row)
    return rows1, rows0


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
       https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return uncertainty - p * gini(left) - (1 - p) * gini(right)

def findSplit(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    bestGain = 0 
    bestQst = None  
    uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  

    for col in range(n_features):  

        values = set([row[col] for row in rows])  

        for val in values:  

            question = Question(col, val)

            
            rows1, rows0 = partition(rows, question)

            if len(rows1) == 0 or len(rows0) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(rows1, rows0, uncertainty)

            if gain > bestGain:
                bestGain, bestQst = gain, question

    return bestGain, bestQst

class Leaf:
    def __init__(self, rows):
        self.predictions = counts(rows)


class Decision_Node:

    def __init__(self,
                 question,
                 branch1,
                 branch0):
        self.question = question
        self.branch1 = branch1
        self.branch0 = branch0


def buildDecisionTree(rows):
    """Building the tree following Google developpers tutorial.
       Rules of recursion:  
            1) Believe that it works.
            2) Start by checking
               for the base case (no further information gain). 
            3) Prepare for giant stack traces.
    """

    gain, question = findSplit(rows)
    if gain == 0:
        return Leaf(rows)

    rows1, rows0 = partition(rows, question)
    branch1 = buildDecisionTree(rows1)
    branch0 = buildDecisionTree(rows0)
    return Decision_Node(question, branch1, branch0)


def split(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return split(row, node.branch1)
    else:
        return split(row, node.branch0)


# Once classified the only true data will be displayed to the farmer to make its decision 
# Improvements : 
#       The farmers will have an account and with the data collected, we can make a recommandation system 
#       and improve the data by recommanding different seeds for all farmers of the country 