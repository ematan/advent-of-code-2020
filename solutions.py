#! /usr/bin/env python3

import sys
import re
import numpy as np
import functools
import time
import itertools
from collections import Counter

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
def day18():
  data = []
  with open("inputs/18", "r") as f:
    data = f.read().split('\n')

  def process(data, math_rule):
    if '(' not in data:
      return math_rule(data)

    parentheses = tuple()
    stack = []

    for i, c in enumerate(data):
      if c == "(":
        stack.append(i)
      elif c == ")":
        parentheses = (stack.pop(), i)
        break


    a, b = parentheses
    e = process(data[a+1: b], math_rule)

    cont = str(data[:a]) + str(e) + str(data[b+1:])
    return process(cont, math_rule)

  def calc(data):
    row = data.split()
    res = int(row[0])
    for s, n in zip(*(iter(row[1:]),) * 2):
      if s == '+':
        res += int(n)
      if s == '*':
        res *= int(n)
    return res

  def calc_2(data):
    parts = data.split('*')
    return np.prod([calc(part) for part in parts])



  print(sum(process(row, calc) for row in data))
  print(sum(process(row, calc_2) for row in data))

#-----------------------------------------------------------------------------------
def day19():
  data = []
  with open("inputs/19", "r") as f:
    data = f.read().split('\n\n')

  rule_dict = {}
  for row in data[0].split('\n'):
    elems = row.split(' ')
    rule_dict[elems[0][:-1]] =  ' '.join(elems[1:])

  def process(d, rule, part=1):
    r = d[rule]
    if r.strip('"') in ['a', 'b']:
      return r.strip('"')

    if part == 2:
      if rule == '8':
        return f'(?:{process(d, "42", part)}+)'
      if rule == '11':
        x = []
        for i in range(1, 20):
          x.append(f'{i * process(d, "42", part)}{i * process(d, "31", part)}')
        return f'(?:{"|".join(x)})'

    res = ''
    for elem in r.split(' '):
      if elem == '|':
        res += elem
      else:
        res += process(d, elem, part)
    return f'(?:{res})'

  p = f'^{process(rule_dict, "0")}$'
  pattern = re.compile(p)

  part1 = sum(1 for row in data[1].split('\n') if pattern.match(row))
  print(part1)


  p = f'^{process(rule_dict, "0", 2)}$'
  pattern = re.compile(p)
  part2 = sum(1 for row in data[1].split('\n') if pattern.match(row))
  print(part2)

#-----------------------------------------------------------------------------------
def day20():
  data = []
  with open("inputs/20", "r") as f:
    data = f.read().strip().split('\n\n')


  class Piece:
    def __init__(self, name, state):
      self._name = name
      self._state = state
      self.top = state[0]
      self.bottom = state[-1]
      self.left = ''.join([row[0] for row in self._state])
      self.right = ''.join([row[-1] for row in self._state])
      self.edges = [self.top, self.right, self.bottom[::-1], self.left[::-1]]
      self.center = [row[1:-1] for row in state[1:-1]]

    def edge_variations(self):
      return self.edges + [e[::-1] for e in self.edges]

    def rotated(self):
      return Piece(self._name, [''.join(t) for t in zip(*reversed(self._state))])

    def flipped(self):
      return Piece(self._name, list(reversed(self._state)))

    def __str__(self):
      return '\n'.join(self._state) + '\n'

    def parse(info):
      x = info.split('\n')
      return Piece(x[0][5:-1], x[1:])

    def __repr__(self):
      return f'Tile({self._name})'


  pieces = [Piece.parse(d) for d in data]

  edge_counts = Counter([e for p in pieces for e in p.edge_variations()])
  corners = []
  edges_per_piece = {}
  for p in pieces:
    if sum(edge_counts[e] for e in p.edges) == 6:
      corners.append(p)
    for e in p.edge_variations():
      edges_per_piece.setdefault(e, []).append(p)


  part1 = np.prod([int(tile._name) for tile in corners])
  print(part1)

  top_left = corners[0].flipped()

  while (edge_counts[top_left.top] != 1 or edge_counts[top_left.left] != 1):
    top_left = top_left.rotated()

  curr = top_left
  puzzle = []
  not_last_row = True
  while(not_last_row):
    row = [curr]
    while(edge_counts[curr.right] != 1):
      x = [p for p in edges_per_piece[curr.right] if p._name != curr._name][0]
      if curr.right[::-1] not in x.edges:
        x = x.flipped()
      while (curr.right != x.left):
        x = x.rotated()
      curr = x
      row.append(curr)
    curr = row[0]
    puzzle.append(row)

    if edge_counts[curr.bottom] == 1:
      not_last_row = False
    else:
      y = [p for p in edges_per_piece[curr.bottom] if p._name != curr._name][0]
      if curr.bottom not in y.edges:
        y = y.flipped()
      n = 0
      while (curr.bottom != y.top and n != 5):
        y = y.rotated()
        n+=1
      curr = y

  completed = [''.join(bb) for row in puzzle for bb in list(zip(*[aa.center for aa in row]))]

  monster = [ '                  # ' ,
              '#    ##    ##    ###' ,
              ' #  #  #  #  #  #   ' ]

  monster_loc = [[i for i in range(len(row)) if row[i]=='#'] for row in monster]

  m_h = len(monster)
  m_w = len(monster[0])
  p_h = len(completed)
  p_w = len(completed[0])

  def monsterAt(picture, i, j, m):
    loc = []
    for y in range(m_h):
      for spot in m[y]:
        if picture[j+y][i+spot] != '#':
          return []
        loc.append((j+y,i+spot))
    return loc


  def locate(picture, m):
    count = 0
    locations = []
    for j in range(p_h-m_h):
      for i in range(p_w-m_w):
        l =  monsterAt(picture, i, j, m)
        if l:
          count +=1
          locations +=l
    return count, locations

  def flipped_world(picture):
    return list(reversed(picture))
  def rotated_world(picture):
    return [''.join(t) for t in zip(*reversed(picture))]

  res = []
  world = completed
  while(len(res) == 0):
    if locate(world, monster_loc)[0] > 0:
      res = world
    elif locate(flipped_world(world), monster_loc)[0] > 0:
      res = flipped_world(world)
    world = rotated_world(world)

  _, loc = locate(res, monster_loc)
  all_hashtags = sum(1 for row in res for i in row if i=='#')

  print(all_hashtags - len(set(loc)))
#-----------------------------------------------------------------------------------
def day21():
  data = []
  with open("inputs/21", "r") as f:
    data = f.read().strip().split('\n')

  pattern = re.compile(r'(.+) \(contains (.+)\)')
  a_dict = {}
  i_list = []
  for row in data:
    i, a = pattern.match(row).groups()
    ingredients = i.split()
    i_list.extend(ingredients)
    for allergen in a.split(', '):
      if allergen in a_dict:
        a_dict[allergen] &= set(ingredients)
      else:
        a_dict[allergen] = set(ingredients)

  def part1():
    i_counts = Counter(i_list)
    not_allergens = i_counts.keys() - set.union(*a_dict.values())
    return sum(i_counts[non] for non in not_allergens)

  def part2():
    confirmed = {}
    unsolved = a_dict.copy()
    while (len(unsolved.keys())> 0):
      best = sorted([len(v),k]for k, v in a_dict.items() if k in unsolved.keys())[0]
      ing = unsolved.pop(best[1])
      confirmed[best[1]] = ing
      for u in unsolved.keys():
        unsolved[u] -= ing

    part2 = [confirmed[key].pop() for key in sorted(confirmed.keys())]
    return ','.join(part2)

  print(part1())
  print(part2())

#-----------------------------------------------------------------------------------
def day22():
  data = []
  with open("inputs/22", "r") as f:
    data = f.read().strip().split('\n\n')
  deck1 = [int(c) for c in data[0].split('\n')[1:]]
  deck2 = [int(c) for c in data[1].split('\n')[1:]]

  def game1(p1, p2):
    while (len(p1)>0 and len(p2)>0):
      c1 = p1.pop(0)
      c2 = p2.pop(0)
      if c1 > c2:
        p1.extend([c1, c2])
      else:
        p2.extend([c2, c1])
    winner = p1 if len(p1)>len(p2) else p2
    return winner

  def score(deck):
    return sum((i+1)*d for i, d in enumerate(reversed(deck)))

  def game2(p1, p2):

    seen = set()
    while(len(p1)>0 and len(p2)>0):
      s = (tuple(p1), tuple(p2))
      if s in seen:
        return 1, [p1, p2]

      seen.add(s)
      c1 = p1.pop(0)
      c2 = p2.pop(0)
      if len(p1) >= c1 and len(p2) >= c2:
        p1_win, _ = game2(p1[:c1], p2[:c2])
        if p1_win:
          p1.extend([c1, c2])
        else:
          p2.extend([c2, c1])
      else:
        if c1 > c2:
          p1.extend([c1, c2])
        else:
          p2.extend([c2, c1])
    winner = len(p1)>len(p2)
    return  winner, [p1,p2]

  print(score(game1(deck1.copy(), deck2.copy())))
  part2 = game2(deck1.copy(), deck2.copy())
  print(score([win for win in part2[1] if len(win)>0][0]))
#-----------------------------------------------------------------------------------
def day23():
  data = []
  with open("inputs/23", "r") as f:
    data = [int(d) for d in  f.read().strip()]

  def game(cups, turns):
    linked = {a:b for a,b in zip(cups, cups[1:]+[cups[0]])}
    curr = cups[0]
    maximum = len(cups)

    for x in range(turns):
      x = curr
      pick_up = [x := linked[x] for _ in range(3)]
      dest = curr
      for i in range(curr-1, curr-5, -1):
        cup = i if i > 0 else maximum+i
        if cup not in pick_up:
          dest = cup
          break
      linked[curr], linked[dest], linked[pick_up[-1]]= linked[pick_up[-1]], linked[curr], linked[dest]
      curr = linked[curr]

    x = 1
    return [x := linked[x] for _ in cups]

  p1 = game(data, 100)
  print(''.join(str(c) for c in p1[:len(data)-1]))

  part2_data = data[:] + [*range(len(data)+1, 1_000_000+1)]
  p2 = game(part2_data,  10_000_000)
  print(p2[0]*p2[1])

#-----------------------------------------------------------------------------------
def day24():
  data = []
  with open("inputs/24", "r") as f:
    data = f.read().strip().split('\n')

  # offset coordinates
  # “odd-r” horizontal layout - shoves odd rows right
  def parse(line):
    q,r = 0, 0
    while line.strip():
      if line[0] == 'w':
        q -= 1
        line = line[1:]
      elif line[0] == 'e':
        q += 1
        line = line[1:]
      elif line[:2] =='se':
        r =  r+1
        line = line[2:]
      elif line[:2] =='sw':
        q, r = q-1, r+1
        line = line[2:]
      elif line[:2] =='ne':
        q, r = q+1, r-1
        line = line[2:]
      elif line[:2] =='nw':
        r = r-1
        line = line[2:]
    return q, r

  counter = Counter(parse(row) for row in data)
  blacks = [x for x, v in counter.items() if v % 2 ]
  print(f'Part 1: {len(blacks)}')

  def neighbors(tile):
    offsets = [(1,-1), (1, 0), (0,1),(-1, 1), (-1,0), (0,-1)]
    n = [tuple(sum(x) for x in zip(o, tile)) for o in offsets if any(o)]
    return n

  def cycle(state):
    new_state = set()
    for tile in state:
      tile_neighbors = neighbors(tile)
      count = sum([1 for n in tile_neighbors if n in state])
      if count in [1, 2]:
        new_state = new_state | {tile}

      potentials = [n for n in tile_neighbors if n not in state]
      for p in potentials:
        if p not in new_state and sum([1 for n in neighbors(p) if n in state]) == 2:
          new_state = new_state | {p}
    return new_state

  world = set(blacks)
  for i in range(1, 101):
    world = cycle(world)
    #print(f'Day {i}: {len(world)}')

  print(f'Part 2: {len(world)}')

#-----------------------------------------------------------------------------------
def day25():
  data = []
  with open("inputs/25", "r") as f:
    data = f.read().strip().split('\n')

  key1, key2 = [int(row) for row in data]
  div = 20201227
  subj = 7

  value = 1
  loopsize = 0
  while value != key2:
    loopsize +=1
    value = subj * value % div

  print(pow(key1, loopsize, div))
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
  try:
    globals()[sys.argv[1]](*sys.argv[2:])
  except KeyboardInterrupt:
    sys.stdout.flush()
    pass