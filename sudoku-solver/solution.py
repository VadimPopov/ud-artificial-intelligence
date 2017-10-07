import sys

assignments = []

rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [row + col for row in A for col in B]

boxes = cross(rows, cols)

row_units = [cross(row, cols) for row in rows]
column_units = [cross(rows, col) for col in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]

# creating a new list for solving diagonal sudoku
diagonals = [[row + col for row, col in zip(rows, cols)], [row + col for row, col in zip(rows, cols[::-1])]]
unitlist = row_units + column_units + square_units + diagonals
units = dict((box, [unit for unit in unitlist if box in unit]) for box in boxes)
peers = dict((box, set(sum(units[box], [])) - set([box])) for box in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board, record it.
    """
    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers

    for unit in unitlist:
        # finding all the boxes in the current unit that have two possible values
        doubles = [box for box in unit if len(values[box]) == 2]
        # comparing all the boxes in the doubles to see if there are identical values
        for i in range(len(doubles)):
            for j in range(i + 1, len(doubles)):
                if values[doubles[i]] == values[doubles[j]]: # twins found!
                    # eliminating twin values for shared peers
                    for box in unit:
                        if box not in (doubles[i], doubles[j]):
                            for num in values[doubles[i]]:
                                assign_value(values, box, values[box].replace(num, ''))
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    return {box: val if val != '.' else cols for box, val in zip(boxes, grid)}

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[box]) for box in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for row in rows:
        print(''.join(values[row + col].center(width) + ('|' if col in '36' else '')
                      for col in cols))
        if row in 'CF': print(line)
    return

def eliminate(values):
    """
    Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    """
    solved = [box for box in values if len(values[box]) == 1]
    for box in solved:
        num = values[box]
        for peer in peers[box]:
            assign_value(values, peer, values[peer].replace(num, ''))
    return values

def only_choice(values):
    """
    Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for unit in unitlist:
        for num in '123456789':
            matches = [box for box in unit if num in values[box]]
            if len(matches) == 1:
                assign_value(values, matches[0], num)
    return values

def reduce_puzzle(values):
    """
    Iterate eliminate() and only_choice(). If at some point, there is a box with no available values, return False.
    If the sudoku is solved, return the sudoku.
    If after an iteration of both functions, the sudoku remains the same, return the sudoku.
    Input: A sudoku in dictionary form.
    Output: The resulting sudoku in dictionary form.
    """
    stuck = False
    while not stuck:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Naked Twins Strategy
        #values = naked_twins(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stuck = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values if len(values[box]) == 0]):
            return False
    return values

def search(values):
    "Using depth-first search and propagation, try all possible values."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False # Failed earlier
    if all(len(values[box]) == 1 for box in values):
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    _, box = min((len(values[box]), box) for box in values if len(values[box]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus
    for value in values[box]:
        new_sudoku = values.copy()
        assign_value(new_sudoku, box, value)
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    return search(grid_values(grid))

if __name__ == '__main__':
    diag_sudoku_grid = sys.argv[1]
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
