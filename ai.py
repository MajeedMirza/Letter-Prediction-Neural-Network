import numpy as np
import re

testing = True #Set to false to input your own string
testStringC = "abcdefghijklmnopqrstuvwxyz "
testStringW = "This is a test to see if this works, this is a test, a test, test"
testPath = "" #If text file path is entered will use this text instead of test string
alphabet = []
alphaSize = 0
inputs = []
inputMatrix = []
npWeights = np.array([])
trainNum = 100 #Used to increase or decrease number of times trained from the same text
pOccurrences = 0 #Only print if the number of occurrences is greater than pOccurrences
readW = True
np.random.seed(123)


def main():
    setCorW()
    inputStr = "  "
    if testing == True:
        if readW == True:
            if testPath != "":
                with open(testPath, 'r') as myfile:
                    inputStr = myfile.read()
            else:
                inputStr = testStringW
        else:
            if testPath != "":
                with open(testPath, 'r') as myfile:
                    inputStr = myfile.read()
            else:
                inputStr = testStringC
        print("Testing is on, therefore learning from predefined string: \n" + inputStr + "\n")
        print("...Reading..." + "\n")
    else:
        inputStr = raw_input("Enter a string to learn from: ")
    global alphabet
    if readW:
        inputStr = re.sub('[^a-zA-Z ]+', '', inputStr.replace("\n"," ")).lower()
        alphabet = constructDictionary(inputStr.replace("\n"," "))
    else:
        inputStr = re.sub('[^A-Za-z0-9]+', '', inputStr).lower()
        alphabet = constructAlphabet(inputStr)
    global alphaSize
    alphaSize = len(alphabet)
    buildMatrices()
    read(inputStr)
    npInputs = np.array(inputMatrix)
    npWeights = 2 * np.random.random((alphaSize, alphaSize)) - 1
    output = train(npInputs, npWeights)
    while True:
        CorWord = "character"
        if readW:
            CorWord = "word"
        inputLetter = raw_input("Enter a " + CorWord + ": ")
        if inputLetter != "":
            printOccurrencesLetter(output, inputLetter, pOccurrences)
            #printOccurrences(output, pOccurences)
            guessNextLetter(output, inputLetter)


def setCorW():
    global readW
    readCorW = raw_input("Enter 'W' to predict words or 'C' to predict characters: ")
    if readCorW.lower() == "c":
        readW = False
    elif readCorW.lower() == "w":
        readW = True
    else:
        print "Please enter either W or C"
        setCorW()


def constructAlphabet(readInput):
    return list(set(readInput))


def constructDictionary(readInput):
    return list(set(readInput.split()))


def buildMatrices():
    global inputs
    global inputMatrix
    for i in xrange(0, alphaSize):
        new = []
        new2 = []
        for j in xrange(0, alphaSize):
            new.append([alphabet[i], alphabet[j]])
            new2.append(0)
        inputs.append(new)
        inputMatrix.append(new2)


def read(readInput):
    learnStr = readInput.lower()
    if readW:
        learnStr = learnStr.split()
    for i in xrange(0, len(learnStr) - 1):
        if learnStr[i] in alphabet and learnStr[i + 1] in alphabet:
            currLetter = learnStr[i]
            nextLetter = learnStr[i + 1]
            row = alphabet.index(currLetter)
            col = alphabet.index(nextLetter)
            inputMatrix[row][col] = inputMatrix[row][col] + 1


def train(npInputs, npWeights):
    for i in xrange(trainNum):
        l0 = npInputs
        l1 = nonlin(np.dot(l0, npWeights))
        l1_error = npInputs - l1
        l1_delta = l1_error * nonlin(l1, True)
        npWeights += np.dot(l0.T, l1_delta)
    return l1


def nonlin(x, drv = False):
    if(drv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def printOccurrences(wghts, greaterThan):
    try:
        for i in xrange(0, alphaSize):
            for j in xrange(0, alphaSize):
                if inputMatrix[i][j] > greaterThan:
                    print('"' + inputs[i][j][0] + '"' + ", " + '"' + inputs[i][j][1] + '"'
                        + " Occurrences: " + str(inputMatrix[i][j]) + ", Probability: " + str(wghts[i][j]))
    except:
        pass


def printOccurrencesLetter(wghts, inputLetter, greaterThan):
    try:
        i = alphabet.index(inputLetter)
        for j in xrange(0, alphaSize):
            if inputMatrix[i][j] > greaterThan:
                print('"' + inputs[i][j][0] + '"' + ", " + '"' + inputs[i][j][1] + '"'
                    + " Occurrences: " + str(inputMatrix[i][j]) + ", Probability: " + str(wghts[i][j]))
    except:
        pass


def guessNextLetter(l1, input):
    currHighest = 0
    guessed = " "
    if input in alphabet:
        i = alphabet.index(input)
        for j in xrange(0, alphaSize):
            if (l1[i][j] * inputMatrix[i][j]) > currHighest:
                currHighest = l1[i][j] * inputMatrix[i][j]
                guessed = inputs[i][j][1]
    else:
        for i in xrange(0, alphaSize):
            for j in xrange(0, alphaSize):
                if (l1[i][j] * inputMatrix[i][j]) > currHighest:
                    currHighest = l1[i][j] * inputMatrix[i][j]
                    guessed = inputs[i][j][1]
    print("Guess: " + '"' + guessed + '"' + "\n")


if __name__ == "__main__":
    main()
