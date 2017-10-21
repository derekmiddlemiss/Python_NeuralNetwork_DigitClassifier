#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:44:12 2017

@author: derekmiddlemiss
"""

from Neural_Network import Neural_Network
import numpy as np

raw_digits = [
          """11111
             1...1
             1...1
             1...1
             11111""",
             
          """.111.
             1...1
             1...1
             1...1
             .111.""",   

          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",
          
          """..1..
             .11..
             ..1..
             ..1..
             11111""",
             
          """.1...
             .1...
             .1...
             .1...
             .1...""",   

          """1....
             1....
             1....
             1....
             1....""",
           
          """...1.
             ...1.
             ...1.
             ...1.
             ...1.""",  

          """....1
             ....1
             ....1
             ....1
             ....1""",  
             
          """11111
             ....1
             11111
             1....
             11111""",

          """1111.
             ....1
             .111.
             1....
             .1111""",

          """11111
             ...1.
             ..1..
             .1...
             11111""",             

          """11111
             ....1
             11111
             ....1
             11111""",

          """1111.
             ....1
             1111.
             ....1
             1111.""",

          """1...1
             1...1
             11111
             ....1
             ....1""",
             
          """...1.
             .1.1.
             11111
             ...1.
             ...1.""",

          """1....
             1.1..
             11111
             ..1..
             ..1..""",                          

          """11111
             1....
             11111
             ....1
             11111""",

          """.1111
             1....
             .111.
             ....1
             1111.""",

          """11111
             1....
             11111
             1...1
             11111""",

          """.1111
             1....
             .111.
             1...1
             .111.""",             

          """11111
             ....1
             ....1
             ....1
             ....1""",

          """11111
             ...1.
             ..1..
             .1...
             1....""",             

          """11111
             1...1
             11111
             1...1
             11111""",

          """.111.
             1...1
             .111.
             1...1
             .111.""",

          """11111
             1...1
             11111
             ....1
             11111""",

          """.111.
             1...1
             .1111
             ....1
             11111""",
             
          """11111
             1...1
             11111
             ....1
             ....1""",

          """.111.
             1...1
             .1111
             ....1
             ....1""",             
             
             ]

def make_digit(raw_digit):
    return [1 if c == '1' else 0 \
            for row in raw_digit.split("\n") \
            for c in row.strip()]

inputs = list(map(make_digit, raw_digits))

targets = [[1,0,0,0,0,0,0,0,0,0],
               [1,0,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,1,0,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],
               [0,0,1,0,0,0,0,0,0,0],               
               [0,0,0,1,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,1,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],               
               [0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,1,0,0],
               [0,0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,1,0],
               [0,0,0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,0,0,0,1]]

target_labels = [0,0,1,1,1,1,1,1,2,2,2,3,3,4,4,4,5,5,6,6,7,7,8,8,9,9,9,9]
    
digit_classifier = Neural_Network(25,10,25)

inputs = np.array( inputs )
targets = np.array( targets )
    
digit_classifier.miniBatchStochasticGD(inputs, targets, 7, 20000)

# =============================================================================
# digit_classifier.batchGD(inputs,targets,10000)
# =============================================================================

def predict(network,input):
    return network.forward_propagate(input)

print("")
print("Training Set")
print("Digit", "Decision Vector")
for i, input in enumerate(inputs):
    outputs = predict(digit_classifier,input)
    print(target_labels[i],"   ", ["%.2f" % p for p in outputs])
    
print("")
print("Test digits:")

print("Sloppy zero")
print("""@@@@.
@...@
@...@
@...@
@@@@.""")
print(["%.2f" % x for x in
          predict( digit_classifier, np.array([1,1,1,1,0,    # @@@@.
                     1,0,0,0,1,    # @...@
                     1,0,0,0,1,    # @...@
                     1,0,0,0,1,    # @...@
                     1,1,1,1,0]))])# @@@@.
print()

print("Stylised three")
print(""".@@@.
...@@
..@@.
...@@
.@@@.""")
print(["%.2f" % x for x in
          predict( digit_classifier, np.array([0,1,1,1,0,    # .@@@.
                     0,0,0,1,1,    # ...@@
                     0,0,1,1,0,    # .@@@.
                     0,0,0,1,1,    # ...@@
                     0,1,1,1,0]))])# .@@@.
print()

print("Stylised eight")
print(""".@@@.
@..@@
.@@@.
@..@@
.@@@.""")
print(["%.2f" % x for x in
          predict( digit_classifier, np.array([0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0,    # .@@@.
                     1,0,0,1,1,    # @..@@
                     0,1,1,1,0]))])# .@@@.
print()
 
print("One with base, no flick")
print("""..@..
..@..
..@..
..@..
.@@@.""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,1,1,1,0]))])# .@@@.
print()

print("One with curved base, no flick")    
print("""..@..
..@..
..@..
..@..
.@.@.""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,1,0,1,0]))])# .@.@.        
print()

print("Squint one")
print("""..@..
..@..
..@..
...@.
...@.""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,1,0,0,    # ..@..
                      0,0,0,1,0,    # ...@.
                      0,0,0,1,0]))]) # ...@.        
print()

print("Sloppy six")       
print("""@@@@.
@....
@@@@.
@...@
@@@@.""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([1,1,1,1,0,    # @@@@.
                      1,0,0,0,0,    # @....
                      1,1,1,1,0,    # @@@@.
                      1,0,0,0,1,    # @...@
                      1,1,1,1,0]))]) # @@@@.   
print()

print("Shifted four")        
print("""....@
..@.@
@@@@@
....@
....@""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([0,0,0,0,1,    # ....@
                      0,0,1,0,1,    # ..@.@
                      1,1,1,1,1,    # @@@@@
                      1,0,0,0,1,    # ....@
                      0,0,0,0,1]))]) # ....@
print()

print("Sloppy four")
print("""@....
@..@.
@@@@.
.@...
.....""")
print(["%.2f" % x for x in
           predict( digit_classifier, np.array([1,0,0,0,0,    # @....
                      1,0,0,1,0,    # @.@..
                      1,1,1,1,0,    # @@@@.
                      0,1,0,0,0,    # ..@..
                      0,0,0,0,0]))]) # .....
print()          
    
    
    
    
    