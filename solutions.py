#! /usr/bin/env python3

import sys
import re
import numpy as np

#day1
def day1():
  with open("inputs/1.txt", "r") as f:
    lines = f.read().split('\n')
    numbers = [int(o) for o in lines]

    results = [i*j for i in numbers for j in numbers if i + j == 2020]
    results2 = [i*j*k for i in numbers for j in numbers for k in numbers if i + j + k == 2020]
    print(results[0])
    print(results2[0])

#day2
def day2():
  pattern = re.compile(r'(\d+)-(\d+) ([a-zA-Z]): ([a-zA-Z]+)')
  count_1 = 0
  count_2 = 0
  with open("inputs/2.txt", "r") as f:
    for line in f:
      a = pattern.match(line)
      (value_1, value_2) = (int(a.group(1)), int(a.group(2)))
      letter = a.group(3)
      password = a.group(4)

      occurences = password.count(letter)
      if value_1 <= occurences <= value_2:
        count_1 +=1

      first = password[value_1 -1] == letter
      second = password[value_2 -1] == letter

      if first != second:
        count_2 +=1

  print(count_1)
  print(count_2)


def day3():
  with open("inputs/3.txt", "r") as f:
    lines = f.read().split('\n')
    slope_width = len(lines[0])
    slope_height = len(lines)

    def count_trees(r, d):
      trees = 0
      for i in range(0, slope_height, d):
        x_pos = (r*i//d)%slope_width

        if lines[i][x_pos] == '#':
          print(i, x_pos)
          trees += 1
      return trees

    print(count_trees(3,1))
    print(np.prod([count_trees(i, j) for i, j in [(1,1),(3,1),(5,1),(7,1),(1,2)]]))




if __name__ == '__main__':
  try:
    globals()[sys.argv[1]](*sys.argv[2:])
  except KeyboardInterrupt:
    sys.stdout.flush()
    pass