import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from Sudoku.search_sudoku_in_image import find_puzzle, get_app_dir, extract_digit
from train_ocr_classifier import load_classifier_model
from sudoku import Sudoku

app_dir = get_app_dir()
IMAGE_PATH = app_dir / 'Images' / 'puzzle_sudoku.jpeg'

if __name__ == '__main__':
    try:
        puzzle, warped = find_puzzle(IMAGE_PATH)

        # initialize our 9x9 Sudoku board
        board = np.zeros((9, 9), dtype="int")
        # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
        # infer the location of each cell by dividing the warped image
        # into a 9x9 grid
        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9
        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cellLocs = []

        # loop over the grid locations
        for y in range(0, 9):
            # initialize the current list of cell locations
            row = []
            for x in range(0, 9):
                # compute the starting and ending (x, y)-coordinates of the
                # current cell
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY
                # add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))

                # crop the cell from the warped transform image and then
                # extract the digit from the cell
                cell = warped[startY:endY, startX:endX]
                digit = extract_digit(cell, debug=False)

                # verify that the digit is not empty
                if digit is not None:
                    # resize the cell to 28x28 pixels and then prepare the
                    # cell for classification
                    roi = cv2.resize(digit, (28, 28))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    # classify the digit and update the Sudoku board with the
                    # prediction
                    model = load_classifier_model()
                    pred = model.predict(roi).argmax(axis=1)[0]
                    print(pred)
                    board[y, x] = pred
            # add the row to our cell locations
            cellLocs.append(row)
        
        # construct a Sudoku puzzle from the board
        print("[INFO] OCR'd Sudoku board:")
        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()
        # solve the Sudoku puzzle
        print("[INFO] solving Sudoku puzzle...")
        solution = puzzle.solve()
        solution.show_full()

    except Exception as e:
        print(e)
