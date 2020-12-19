#! /usr/bin/env python3

import sys
import re
import numpy as np
import functools
import time
import itertools

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
    for c in input:
      if c == 'F':
        hi -= (hi-lo+1)/2
      if c == 'B':
        lo += (hi-lo+1)/2
    return int(lo)

  def col(input):
    l, r = 0, 7
    for c in input:
      if c == 'L':
        r -= (r-l+1)/2
      if c == 'R':
        l += (r-l+1)/2
    return int(l)

  def seatID(input):
    return 8*row(input[:7])+col(input[7:])

  with open("inputs/5", "r") as f:
    lines = f.read().split()

    minId = min([seatID(line) for line in lines])
    maxId = max([seatID(line) for line in lines])
    ids = {seatID(line) for line in lines}
    allIds = set(range(minId, maxId+1))
    print(maxId)
    print(max(allIds-ids))

#-----------------------------------------------------------------------------------
def day6():
  with open("inputs/6", "r") as f:
    groups = f.read().split('\n\n')

    part1 = sum([len(set(group.replace('\n', ''))) for group in groups])
    print(part1)
    rows = [[set(row) for row in group.split('\n')] for group in groups]
    part2 = sum([len(set.intersection(*row)) for row in rows])
    print(part2)

#-----------------------------------------------------------------------------------
def day7():
  d = {}
  pattern = re.compile(r'(\d+?) (.+?) bag+')

  with open("inputs/7", "r") as f:
    lines = f.read().split('\n')

    for line in lines:
      (l1, l2) = line.split(' bags contain')
      inner = pattern.findall(l2)
      inner_dict = dict()
      for i in inner:
        inner_dict[i[1]] = int(i[0])
      d[l1] = inner_dict

  def rec_part1(bags):
    for bag in bags:
      if bag == 'shiny gold':
        return True
      else:
        if rec_part1(d[bag]):
          return True
    return False

  gold_capacities = {x:rec_part1(d[x]) for x in d}
  part1 = len({x for x in gold_capacities if (gold_capacities[x]==True)})
  print(part1)


  @functools.lru_cache(maxsize=None)
  def rec_part2(bagname):
    if d[bagname] == {}:
      return 0
    else:
      count = 0
      for key, value in d[bagname].items():
        count += value * (1+rec_part2(key))
      return count

  print(rec_part2('shiny gold'))

#-----------------------------------------------------------------------------------
def day8():
  program = []
  flags = []
  flippable = []

  with open("inputs/8", "r") as f:
    lines = f.read().split('\n')
    program = [line.split() for line in lines]
    flags = [0] * len(lines)

  def process_row(data, i, acc):
    if i >= len(data):
      return (acc, True)
    elif flags[i] == 1:
      return (acc, False)
    else:
      flags[i] = 1
      command, value = data[i]
      if command == 'acc':
        return process_row(data, i+1, acc + int(value))
      if command == 'jmp':
        return process_row(data, i+int(value), acc)
      if command == 'nop':
        return process_row(data, i+1, acc)

  part1, _ = process_row(program, 0, 0)
  print('Part1:', part1)

  for i in range(0, len(program)):
    if program[i][0] in ('jmp', 'nop'):
      flippable.append(i)

  for i in flippable:
    program[i][0] = 'nop' if program[i][0] == 'jmp' else 'jmp'
    flags = [0] * len(program)
    part2, success = process_row(program, 0, 0)
    if success:
      print('Part2:', part2)
      break
    program[i][0] = 'nop' if program[i][0] == 'jmp' else 'jmp'

#-----------------------------------------------------------------------------------
def day9():
  rows = []

  def read_data():
    with open("inputs/9", "r") as f:
      data = f.read().split('\n')
      data = [int(i) for i in data]
      return data

  def check(arr, value):
    hit = [i for i in arr for j in arr if i+j==value]
    return len(hit)==0

  invalid = 0
  rows = read_data()
  preamble = rows[0:25]
  rest = rows[25:len(rows)]

  while len(rest)>0:
    current = rest.pop(0)
    if check(preamble, current):
      print('Part1:' , current)
      invalid = current
      break
    preamble.pop(0)
    preamble.append(current)


  def part2(arr, target):
    i = 0
    while i < len(arr):
      j = i
      contiguous_sum = 0
      while j < len(arr) and contiguous_sum < target:
        contiguous_sum += arr[j]
        if contiguous_sum == target:
          return (i,j)
        j += 1
      i += 1
    return (0,0)

  rows = read_data()
  i, j = part2(rows, invalid)
  arr = rows[i:(j+1)]

  print('Part2:', min(arr)+max(arr))

#-----------------------------------------------------------------------------------
def day10():
  with open("inputs/10", "r") as f:
    data = f.read().split('\n')
    numbers = [int(i) for i in data]
    goal = max(numbers)+3
    numbers.extend((0, goal))

    data_s = sorted(numbers, reverse=True)
    l = len(data_s)

    diff = [data_s[i] - data_s[i+1] for i in range(0,l-1)]
    part1 = diff.count(3) * diff.count(1)
    print('Part1:',part1)

    data_s = sorted(numbers)
    solutions_to_here = {0: 1}
    for joltage in data_s:
      count = 0
      for i in range(0,4):
        if (joltage-i) in solutions_to_here:
          count += solutions_to_here[joltage-i]
      solutions_to_here[joltage] = count

    print('Part2:',solutions_to_here[goal])

#-----------------------------------------------------------------------------------
def day11():
  with open("inputs/11", "r") as f:
    data = f.read().split('\n')
    width = len(data[0])
    height = len(data)

    state = data

    def part1_adjacents(state, x, y):
      minX = x-1 if x-1>=0 else x
      maxX = x+1 if x+1<width else x
      minY = y-1 if y-1>=0 else y
      maxY = y+1 if y+1<height else y

      count = 0
      for i in range(minX, maxX+1):
        for j in range(minY, maxY+1):
          if state[j][i] == '#' and not (i==x and j==y):
            count +=1
      return count

    def part1_evolution(state, i,j):
      c = part1_adjacents(state, i, j)
      if state[j][i] == 'L' and c==0:
        return '#'
      elif state[j][i] == '#' and c>3:
        return 'L'
      else:
        return state[j][i]

    def part2_adjacents(state, x, y, dirX, dirY):
      curX, curY = x+dirX, y+dirY
      while(0<=curX<width and 0<=curY<height):
        if state[curY][curX]=='#':
          return 1
        elif state[curY][curX]=='L':
          return 0
        else:
          curX += dirX
          curY += dirY
      return 0

    DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

    def part2_evolution(state, i, j):
      c = 0
      for dir in DIRS:
        c += part2_adjacents(state, i, j, dir[0], dir[1])
      if state[j][i] == '#' and c > 4:
        return 'L'
      if state[j][i] == 'L' and c == 0:
        return '#'
      else:
        return state[j][i]

    def evolve(world, evolve_func):
      no_diff = True
      while(no_diff):
        newworld = []
        for j in range(0,height):
          row = ''
          for i in range(0, width):
            row += evolve_func(world, i, j)
          newworld.append(row)
        if (world == newworld):
          no_diff=False
        world=newworld
        #time.sleep(0.1)
        #print('-------------------')
        #print('\n'.join(world))
      return world

    s = evolve(state, part1_evolution)
    part1 = sum(row.count('#') for row in s)
    print(part1)
    s2 = evolve(state, part2_evolution)
    part2 = sum(row.count('#') for row in s2)
    print(part2)

#-----------------------------------------------------------------------------------
def day12():

  class Boat:
    def __init__(self):
      self._positionX = 0
      self._positionY = 0
      self._direction = 'E'
      self._actions = {
        'N': lambda y: self.move(0, y),
        'E': lambda x: self.move(x, 0),
        'S': lambda y: self.move(0, -y),
        'W': lambda x: self.move(-x, 0),
        'L': lambda deg: self.turn(-deg),
        'R': lambda deg: self.turn(deg),
        'F': lambda val: self._actions[self._direction](val)
      }

    def move(self, x, y):
      self._positionX +=x
      self._positionY +=y

    def turn(self, deg):
      dirs = 'NESW'
      new_index = (dirs.index(self._direction) + (deg // 90)) % 4
      self._direction = dirs[new_index]

    def do(self, action, value):
      self._actions[action](value)

    def manhattan(self):
      return abs(self._positionX) + abs(self._positionY)


  class Waypoint:
    def __init__(self):
      self._positionX = 10
      self._positionY = 1
      self._boat = [0, 0]
      self._actions = {
        'N': lambda y: self.move(0, y),
        'E': lambda x: self.move(x, 0),
        'S': lambda y: self.move(0, -y),
        'W': lambda x: self.move(-x, 0),
        'L': lambda deg: self.turn(-deg),
        'R': lambda deg: self.turn(deg),
        'F': lambda val: self.move_boat(val)
      }

    def move_boat(self, value):
      self._boat[0] += value * self._positionX
      self._boat[1] += value * self._positionY

    def move(self, x, y):
      self._positionX +=x
      self._positionY +=y

    def turn(self, deg):
      x = (deg // 90) % 4
      if x == 1:
        self._positionX, self._positionY = self._positionY, -self._positionX
      elif x == 2:
        self._positionX, self._positionY = -self._positionX, -self._positionY
      elif x == 3:
        self._positionX, self._positionY = -self._positionY, self._positionX

    def do(self, action, value):
      self._actions[action](value)

    def manhattan(self):
      return abs(self._boat[0]) + abs(self._boat[1])

  with open("inputs/12", "r") as f:
    data = f.read().split('\n')

    boatyMcBoatFace = Boat()
    totallyLegitDirection = Waypoint()
    pattern = re.compile(r'([A-Z])(\d+)')
    for row in data:
      action, value = pattern.match(row).groups()
      boatyMcBoatFace.do(action, int(value))
      totallyLegitDirection.do(action, int(value))
    print(boatyMcBoatFace.manhattan())
    print(totallyLegitDirection.manhattan())


#-----------------------------------------------------------------------------------
def day13():
  data = []
  with open("inputs/13", "r") as f:
    data = f.read().split('\n')
    timestamp = int(data[0])
    buses = [int(b) for b in data[1].split(',') if b!='x']
    busdict = {}
    for bus in buses:
      busdict[bus] = (bus - timestamp) % bus
    best = min(busdict, key=busdict.get)
    part1 = best * busdict[best]
    print('Part1:', part1)

    multiplier = 1
    result = 0
    indexed_buses = [(x[0], int(x[1])) for x in enumerate(data[1].split(',')) if x[1]!='x']
    for i, bus in indexed_buses:
      while (result + i) % bus != 0:
        result += multiplier
      multiplier *= bus

    print('Part2:',result)
#-----------------------------------------------------------------------------------
def day14():
  data = []
  with open("inputs/14", "r") as f:
    data = f.read().split('\n')

  pattern = re.compile(r'mem\[(\d+)\]')

  def part1(lines):
    mask = 'X'*36
    all_memories = {}
    for row in lines:
      op, _, value = row.split(' ')
      if op == 'mask':
        mask = value
      else:
        temp_mask = list(mask)
        slot = int(pattern.match(op).groups()[0])
        value_in_bit = f"{int(value):036b}"

        for i in range(0, len(temp_mask)):
          if temp_mask[i] == 'X':
            temp_mask[i] = value_in_bit[i]
        all_memories[slot] = int(''.join(temp_mask), 2)
    return sum(all_memories.values())

  def part2(lines):
    mask = 'X'*36
    all_memories = {}
    for row in lines:
      op, _, value = row.split(' ')
      if op == 'mask':
        mask = value
      else:
        temp_mask = list(mask)
        slot = int(pattern.match(op).groups()[0])
        slot_in_bit = f"{slot:036b}"

        floaters = []
        for i in range(0, len(mask)):
          if temp_mask[i] == '0':
            temp_mask[i] = slot_in_bit[i]
          if temp_mask[i] == 'X':
            floaters.append(i)

        for ind in range(0, np.power(2, len(floaters))):
          current = temp_mask.copy()
          index_in_bit = str(bin(ind))[2:]
          index_in_bit = '0' * (len(floaters) - len(index_in_bit)) + index_in_bit
          for i, j in zip(floaters, index_in_bit):
            current[i] = j
          all_memories[''.join(current)] = int(value)
    return sum(all_memories.values())


  print(part1(data))
  print(part2(data))

#-----------------------------------------------------------------------------------
def day15():
  data = []
  with open("inputs/15", "r") as f:
    data = [int(d) for d in f.read().split(',')]

  roundsA=2020
  roundsB=30000000

  def run(year):
    mem = {val:i for i, val in enumerate(data[:len(data)-1])}
    prev = data[-1]

    for i in range(len(data), year):
      if prev in mem:
        cur=i-mem[prev]-1
      else:
        cur = 0
      mem[prev] = i-1
      prev = cur
    return prev

  print(run(roundsA))
  print(run(roundsB))

#-----------------------------------------------------------------------------------
def day16():
  data = []
  with open("inputs/16", "r") as f:
    data = f.read().split('\n\n')


  field_values = dict()
  pattern = re.compile(r'(.+): (\d+)-(\d+) or (\d+)-(\d+)')
  for l in data[0].split('\n'):
    op, x1, x2, y1, y2 = pattern.match(l).groups()
    field_values[op] = set([*range(int(x1), int(x2)+1)] + [*range(int(y1), int(y2)+1)])

  ticket = [int(x) for x in data[1].split('\n')[1].split(',')]
  others = [[int(y) for y in x.split(',')] for x in data[2].split('\n')[1:]]

  field_dict = {i:set(field_values.keys()) for i in range(len(field_values))}
  valid_numbers = set.union(*field_values.values())

  # Part1
  part1 = sum([value for row in others for value in row if value not in valid_numbers])
  print(part1)


  # Part2
  good = [row for row in others if all([x in valid_numbers for x in row])]
  for row in good:
    for i in range(len(row)):
      possibles = field_dict[i]
      impossibles = set()
      for name in possibles:
        if row[i] not in field_values[name]:
          impossibles = impossibles | {name}
      field_dict[i] = possibles - impossibles


  solved = {}
  unsolved = set([*range(len(ticket))])
  while(len(unsolved)>0):
    best = sorted([[len(v),k] for k, v in field_dict.items() if k in unsolved])[0]
    if best[0] == 1:
      unsolved = unsolved - {best[1]}
      name = field_dict[best[1]]
      solved[best[1]] = name
      for u in unsolved:
        field_dict[u] -= name

  part2 = [ticket[k] for k, v in field_dict.items() for val in v if 'departure' in val]
  print(np.prod(part2))

#-----------------------------------------------------------------------------------
def day17():
  data = []
  with open("inputs/17", "r") as f:
    data = f.read().split('\n')

  def initialize(dim):
    return set((j,i)+(0,)*(dim-2) for i in range(len(data)) for j in range(len(data[0])) if data[i][j] == '#')

  def neighbors(point):
    offsets = itertools.product([-1, 0, 1], repeat=len(point))
    n = [tuple(sum(x) for x in zip(o, point)) for o in offsets if any(o)]
    return n

  def cycle(state):
    new_state = set()
    for cube in state:
      cube_neighbors = neighbors(cube)
      count = sum([1 for n in cube_neighbors if n in state])
      if count in [2,3]:
        new_state = new_state | {cube}

      potentials = [n for n in cube_neighbors if n not in state]
      for p in potentials:
        if p not in new_state and sum([1 for n in neighbors(p) if n in state]) == 3:
          new_state = new_state | {p}
    return new_state

  def run(dim):
    state = initialize(dim)
    for _ in range(6):
      state = cycle(state)
    return len(state)

  print(run(3))
  print(run(4))

#-----------------------------------------------------------------------------------
if __name__ == '__main__':
  try:
    globals()[sys.argv[1]](*sys.argv[2:])
  except KeyboardInterrupt:
    sys.stdout.flush()
    pass