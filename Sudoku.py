import matplotlib.pyplot as plt
import copy

class Sudoku:

    def __init__(self, A):
        self.S = A
        self.find_neighbors()

    def draw(self, title='', show_rc_nums=False, show_valid_vals=False):
        # Draw lines
        fig, self.ax = plt.subplots(figsize=(8,8))
        for i in range(0,10,3):
            self.ax.plot([i,i],[0,9],linewidth=2,color='k')
            self.ax.plot([0,9],[i,i],linewidth=2,color='k')
        for i in range(1,9):
            self.ax.plot([i,i],[0,9],linewidth=1,color='k')
            self.ax.plot([0,9],[i,i],linewidth=1,color='k')

        # Print row and column numbers if desired
        if show_rc_nums:
            for i in range(9):
                self.ax.text((-.5),(i+.5), str(i), size=12,color = 'r',
                    ha="center", va="center")
                self.ax.text((i+.5),(-.5), str(i), size=12,color = 'r',
                    ha="center", va="center")

        # Print known values
        for i in range(9):
            for j in range(9):
                if self.S[i][j] != 0:
                    self.ax.text((j+.5),(i+.5), str(self.S[i][j]), size=18,
                        ha="center", va="center")

        # Print valid values using small green numbers, if desired
        if show_valid_vals and hasattr(self,'V'):
            for i in range(9):
                for j in range(9):
                    if self.S[i][j] == 0:
                        for n in self.V.get((i,j), []):
                            n1 = n-1
                            self.ax.text((j+.5+(n1%3-1)*.25),(i+.5+(n1//3-1)*.25), str(n), size=10,
                                         color = 'g', ha="center", va="center")

        self.ax.axis('off')
        self.ax.set_title(title, y=-.05,size = 18)
        self.ax.set_aspect(1.0)
        self.ax.invert_yaxis()
        plt.show()

    def find_neighbors(self):
        # self.N[(r, c)] contains a set of coordinate tuples (i, j) of all cells
        # that are in the same row, column, or 3x3 subgrid as the cell (r, c), excluding (r, c) itself.
        # For example, for cell (0, 0), self.N[(0, 0)] includes all cells in row 0,
        # all cells in column 0, and all cells in the top-left 3x3 subgrid, except for (0, 0).
        # This dictionary will be used to quickly access neighboring cells when checking for constraints.
        self.N = {}

        #Traverse sudoku board
        for r in range(9):
          for c in range(9):
            self.N[(r, c)] = self.find_neighbors_helper(r, c) #Use helper method to find (r, c)'s neighbors

        return self.N

    def find_neighbors_helper(self, r, c):

      neighbors = set() #Create set for (r, c)'s neighbors

      #Fill out with tuples on the same row
      for i in range(9):
          if i != c:
            neighbors.add((r, i))

      #Fill out with tuples on the same column
      for i in range(9):
        if i != r:
          neighbors.add((i, c))

      #Find starting index for the row of the 3x3 block (r, c) is in by finding int division result, and then multiplying by 3
      start_row = (r // 3) * 3
      #Find starting index for the column of the 3x3 block (r, c) is in by finding int division result, and then multiplying by 3
      start_column = (c // 3) * 3

      #Iterate through the 3x3 block by starting with the starting row and column found before and stopping 2 blocks after (delimited using variable + 3)
      for m in range(start_row, start_row + 3):
        for n in range(start_column, start_column + 3):
          if (m, n) != (r, c): #Make sure that (r, c) itself is not being added to the neighbors set
            neighbors.add((m, n)) #Otherwise, add the coordinate to the set

      return neighbors #Return the set with (r, c)'s neighbors

    def init_valid(self):
        # Using the neighbor dictionary self.N from find_neighbors(), self.V is filled out
        # For each cell (r, c), self.V[(r, c)] contains the set of valid numbers (1-9) that can be placed in that cell
        # without violating Sudoku rules (i.e., no duplicates in the same row, column, or 3x3 subgrid).
        # If a number is already placed in cell (r, c) in self.S (i.e., self.S[r][c] != 0),
        # then self.V[(r, c)] should be an empty set since the cell's value is fixed.
        # For empty cells, the set is computed by excluding numbers that appear in any of its neighboring cells in self.N[(r, c)].
        self.V = {}

        #Traverse sudoku board
        for r in range(9):
          for c in range(9):
            if self.S[r][c] != 0: #If value is already fixed, then the set of the valid values is empty
              self.V[(r, c)] = set()
            else:
              self.V[(r, c)] = self.helper_init_valid(r, c) #Otherwise, use helper method to find valid numbers for (r, c)

    def helper_init_valid(self, r, c):

      #Initialize valid_set with all possible values 1-9
      valid_set = set([1, 2, 3, 4, 5, 6, 7, 8, 9])

      #Traverse (r, c)'s neighbors
      for coordinate in self.N[(r, c)]:
        #Create a list that will store the value of the board at the current coordinate in (r, c)'s neighbors
        numbers_list = []
        #Parse coordinate to a list so we can access row and column individually
        new_list = list(coordinate)
        #Parse list[0] (which is the row) to an int so we can use it as an index to access element in sudoku board
        row = int(new_list[0])
        #Parse list[1] (which is the column) to an int so we can use it as an index to access element in sudoku board
        column = int(new_list[1])

        #Add the element at the current coordinate to numbers_list
        numbers_list.append(self.S[row][column])

        #Check if (r, c)'s neighbor's value is in valid set. If it is, remove it from valid_set since it would be an invalid operation to use it in
        #(r, c)
        if numbers_list[0] in valid_set:
          valid_set.remove(numbers_list[0])

      #Once all of (r, c)'s neighbors are traversed, return (r, c)'s valid numbers set
      return valid_set

    def solve(self):

      self.init_valid() #Initialize init_valid()

      #known set will store a tuple in the form (val, r, c) if the coordinate (r, c) only has 1 valid value, whoch is represented by val
      known = set()

      #Fill known set with all the coordinates that only have 1 valid value
      for r in range(9):
        for c in range(9):
          if len(self.V[(r, c)]) == 1: #Check if the valid set for (r, c) is of length 1
            val = self.V[(r, c)].pop() #If it is, pop the value
            known.add((val, r, c)) #Add this value to known, as well as its coordinates all in the same tuple

      #Do this as long as there is something in known
      while len(known) > 0:
        #pop a random tuple from known set
        (val, r, c) = known.pop()
        #Since we know that the value in this tuple is the only possible value for (r, c), we can set S[r][c] to be val
        self.S[r][c] = val
        #The valid set for (r, c) will now be empty since we've fixed a value for this coordinate
        self.V[(r, c)] = set()

        #Traverse (r, c)'s neighbors so we can remove val from its neighbors' valid sets
        for (i, j) in self.N[(r, c)]:
          #Check if val is in the neighbor's valid set
          if val in self.V[(i, j)]:
            #If it is, remove val from valid set
            self.V[(i, j)].remove(val)

        #Traverse the sudoku board again and check for new cells that only have 1 valid value. This does the same as the two for loops before
        #the while loop; maybe I could have implemented a helper method that only does this check and that can update known set
        for m in range(9):
          for n in range(9):
            if len(self.V[(m, n)]) == 1:
              valid = self.V[(m, n)].pop()
              known.add((valid, m, n))

        #This is important so we can be sure there's no duplicates (which was happening to me in the last easy puzzle). This is a little bit hard to
        #explain without showing an example, but the thing is that if we have a tuple in known whose value was previously unique to that cell, but later on
        #we fill one of its neighbors with that same value, then it means that the original cell should have an empty set and there is nothing else
        #left to put in there. This is the case when we want to return -1
        for value in known: #Traverse known set
          (val2, r2, c2) = value #Check one of the tuples
          #If this tuple's value is the same as val and its coordinates are in (r, c)'s neighbors, then the scenario mentioned above is happening
          if val2 == val and (r2, c2) in self.N[(r, c)]:
            return -1 #Therefore, we return 0 since the cell is unfilled and its valid values set is empty

      #Counter to check if all the cells in the board are filled
      count_filled_cells = 0

      #Traverse the board
      for r in range(9):
        for c in range(9):
          if self.S[r][c] != 0: #If the cell's value is not 0, add 1 to the counter
            count_filled_cells += 1

      #If all of the 81 cells are filled, then it means that the puzzle was successfully completed and we can return 1
      if count_filled_cells == 81:
        return 1

      #Otherwise, return 0
      return 0

    def solve_backtrack(self):

      #Initialize sol by using self.solve(), this will make it so solve does everything it can until there are no more single possible values
      sol = self.solve()

      #If sol is 1, then the puzzle is solvable and 1 is returned
      if sol == 1:
        return 1
      #If sol is -1, the puzzle is unsolvable
      if sol == -1:
        return -1

      #Use helper method to find an empty cell to start working on
      (r, c) = self.solve_backtrack_find_empty()

      #Traverse (r, c)'s set of valid values
      for value in self.V[(r, c)]:

        #Make a copy of the original board so we can go back to it if the backtrack doesn't work
        original_board = copy.deepcopy(self.S)
        original_V = copy.deepcopy(self.V)

        #Start backtrack by choosing a value from the valid values set for (r, c) and trying to go from there
        self.S[r][c] = value

        #Traverse (r, c)'s neighbors coordinates and check if value is in any of their valid sets
        for (i, j) in self.N[(r, c)]:
          if value in self.V[(i, j)]:
            self.V[(i, j)].remove(value) #If it is, remove it

        #Call solve_backtrack again and see if the guess works
        sol = self.solve_backtrack()

        #If sol is 1, the puzzle is solved, return 1
        if sol == 1:
          return 1

        #Go back to the original board if backtrack doesn't work
        self.S = copy.deepcopy(original_board)
        self.V = copy.deepcopy(original_V)

      #Return None if solution is not found
      return None

    def solve_backtrack_find_empty(self): #Helper method to find empty cell

      #Best will be the coordinate with the least amount of possible valid values
      best = None
      #Since the valid values are at most 1-9, we can set best_len to be anything greater than 9 to compare
      best_len = 10

      for r in range(9):
        for c in range(9):
            if self.S[r][c] == 0: #Find an empty cell
                L = len(self.V[(r, c)])
                #This case shouldn't happen since it would be an error to have an empty cell that also has no possible values
                if L == 0:
                    return (-1, -1)
                #Update best_len and best cell if we find a cell with less possible valid values than the last best
                if L < best_len:
                    best_len = L
                    best = (r, c)
      return best  #best will be None if the puzzle is solved

#Easy puzzles with simple solve   
f = open("easy21.txt", "r")
count = [0,0,0]
ss = [s for s in f.read().split('\n')]


def convert_to_board(s):
  board = []
  counter = 0
  row = []
  for c in s:
    if counter == 9:
      board.append(row)
      row = []
      counter = 0
    if c != '.':
      row.append(int(c))
    else:
      row.append(0)
    counter += 1
  if row:
    board.append(row)
  return board

for s in ss:
  board = convert_to_board(s)
  S = Sudoku(board)
  S.find_neighbors()
  S.init_valid()
  S.draw()
  S.draw(show_rc_nums=True, show_valid_vals=True)

  sol = S.solve()
  S.draw()
  print(sol)
  print()
  count[sol]+=1

print('solved puzzles:',count[1])
print('unsolved puzzles:',count[0])
print('unsolvable puzzles:',count[-1])


#Easy puzzles solved with backtracking
f = open("easy21.txt", "r")
count = [0,0,0]
ss = [s for s in f.read().split('\n')]
print(len(ss), 'strings read')
for s in ss:
  board = convert_to_board(s)
  sudoku = Sudoku(board)
  sudoku.draw()
  sol = sudoku.solve_backtrack()
  sudoku.draw()
  print(sol)
  count[sol]+=1

print('solved puzzles:',count[1])
print('unsolved puzzles:',count[0])
print('unsolvable puzzles:',count[-1])

#Hard puzzles solved with backtracking
import time
f = open("hard1000.txt", "r")
count = [0,0,0]
ss = [s for s in f.read().split('\n')]
print(len(ss), 'strings read')
start = time.time()
for s in ss[:10]: 
  board = convert_to_board(s)
  sudoku = Sudoku(board)
  sudoku.draw()
  sol = sudoku.solve_backtrack()
  sudoku.draw()
  print('solve_backtrack(s) result:\n',sol)
  if sol!=None:
    count[1]+=1
  else:
    count[-1]+=1
elapsed_time2 = time.time() - start
print('elapsed time using set', elapsed_time2,'secs')

print('solved puzzles:',count[1])
print('unsolved puzzles:',count[0])
print('unsolvable puzzles:',count[-1])
