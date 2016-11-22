# This example is from the simpleai library document available at
# http://simpleai.readthedocs.io/en/latest/#installation and written
# in my own personal format to assist with understanding.

# The example itself attempts to create the string HELLO WORLD using
# the A* algorthim(pronounced "A star".  Used most common for applying
# a informed search or best-first search to solve problems.
# You can learn more about # it at is's wikipedia.org web page.
# https://en.wikipedia.org/wiki/A*_search_algorithm

from simpleai.search import SearchProblem, astar

GOAL = 'HELLO WORLD'

class Hello(SearchProblem):
    def actions(self, state):
        if len(state) < len(GOAL):
            return list(' ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            return []

    def result(self, state, action):
        return state + action

    def is_goal(self, state):
        return state == GOAL

    def heuristic(self, state):
        # how far from goal
        wrong = sum([1 if state[i] != GOAL[i] else 0
                     for i in range(len(state))])
        missing = len(GOAL) - len(state)
        return wrong + missing

problem = Hello(initial_state='')
result = astar(problem)

print (result.state())
print (result.path())
