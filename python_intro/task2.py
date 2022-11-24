import random

random_number = random.randint(0, 9)

tries = []
while True:
  x = int(input("What is your guess?"))
  tries.append(x)
  if x==random_number:
    print("Number of tries", len(tries))
    print(tries)
    break
