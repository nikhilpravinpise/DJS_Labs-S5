## Department of Computer Science and Engineering (Data Science)

## Subject: Artificial Intelligence (DJ19DSC502)

AY: 2026-27

## Experiment 1

## (Problem Solving)

Aim: Implement domain specific functions for given problems required for problem solving.

## Theory:

There are two domain specific functions required in all problem solving methods.

- 1. GoalTest Function:

goalTest(State) Returns true if the input state is the goal state and false otherwise.

goalTest(State, Goal) Returns true if State matches Goal, and false otherwise.

- 2. MoveGen function:

Initialize set of successors C to empty set. Add M to the complement of given state N to get new state S. If given state has Left, then add Right to S, else add Left. If legal(S) then add S to set of successors C. For each other-entity E in N

make a copy S’ of S, add E to §’, If legal (S’), then add S’ to C.

Return (C).

## Lab Assignment to do:

Create MoveGen and GoalTest Functions for the given problems

- 1. Water Jug Problem

There are three jugs available of different volumes such as a 3 litres and a 7 litres and you have to measure a different volume such as 6 litre.

## 2. Travelling Salesman Problem

A salesman is travelling and selling his/her product to in different cities. The condition is that it has to travel each city just once.

- 3. 8 Puzzle Problem

An initial state is given in a 8 puzzle where one place is blank out of 9 places. You can shift this blank space and get a different state to reach to a given goal state.
