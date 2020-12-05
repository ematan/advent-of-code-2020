#! /usr/bin/env python3

import sys
import re
import numpy as np

#-----------------------------------------------------------------------------------
def day1():
  with open("inputs/1.txt", "r") as f:
    lines = f.read().split('\n')
    numbers = [int(o) for o in lines]

    results = [i*j for i in numbers for j in numbers if i + j == 2020]
    results2 = [i*j*k for i in numbers for j in numbers for k in numbers if i + j + k == 2020]
    print(results[0])
    print(results2[0])

#-----------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------
def day4():
  req_fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}

  def hgt_validation(value):
    pattern = re.compile(r'(\d*)(in|cm)')
    data = pattern.match(value)
    if data:
      h, unit = data.groups()
      return ((unit == 'cm' and 150 <= int(h) <= 193) or
              (unit == 'in' and  59 <= int(h) <= 76))
    return False

  validations = {
    'byr': (lambda byr: 1920 <= int(byr) <= 2002),
    'iyr': (lambda iyr: 2010 <= int(iyr) <= 2020),
    'eyr': (lambda eyr: 2020 <= int(eyr) <= 2030),
    'hgt': (lambda hgt: hgt_validation(hgt)),
    'hcl': (lambda hcl: (re.compile(r'#[\da-z]{6}').match(hcl) is not None) ),
    'ecl': (lambda ecl: ecl in ['amb','blu', 'brn', 'gry', 'grn', 'hzl', 'oth']),
    'pid': (lambda pid: (len(pid)==9 and pid.isnumeric())),
  }

  def validate(data):
    for f in req_fields:
      if not f in data or not (validations[f](data[f])):
        return False
    return True

  def part1(passports):
    return sum([all(f in p for f in req_fields) for p in passports])

  def part2(passports):
    formatted = [x.split() for x in passports]
    formatted = [dict([y.split(':') for y in x]) for x in formatted]
    return sum([validate(x) for x in formatted])

  with open("inputs/4", "r") as f:
    input = f.read().split('\n\n')

    print(part1(input))
    print(part2(input))

#-----------------------------------------------------------------------------------
def day5():
  def row(input):
    lo, hi = 0, 127
    for i in range(0,7):
      if input[i] == 'F':
        hi -= (hi-lo+1)/2
      if input[i] == 'B':
        lo += (hi-lo+1)/2
    return int(lo)

  def col(input):
    l, r = 0, 7
    for i in range(7,10):
      if input[i] == 'L':
        r -= (r-l+1)/2
      if input[i] == 'R':
        l += (r-l+1)/2
    return int(l)

  def seatID(input):
    return 8*row(input)+col(input)

  with open("inputs/5", "r") as f:
    lines = f.read().split()

    minId = min([seatID(line) for line in lines])
    maxId = max([seatID(line) for line in lines])
    ids = {seatID(line) for line in lines}
    allIds = set(range(minId, maxId+1))
    print(maxId)
    print(max(allIds-ids))







#-----------------------------------------------------------------------------------
if __name__ == '__main__':
  try:
    globals()[sys.argv[1]](*sys.argv[2:])
  except KeyboardInterrupt:
    sys.stdout.flush()
    pass