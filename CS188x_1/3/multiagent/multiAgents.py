# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # print bestScore, "bestScore"
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        oldFood = currentGameState.getFood()
        foodEatingIncentiveScore = 5*(oldFood.count() - newFood.count())
        newFoodDistances = map(lambda x: manhattan_distance(x, newPos), newFood.asList())
        oldFoodDistances = map(lambda x: manhattan_distance(x, currentGameState.getPacmanPosition()), oldFood.asList())
        moveTowardFoodScore = 0.0
        if oldFood.count() - newFood.count() == 0 and min(newFoodDistances) < min(oldFoodDistances):
            moveTowardFoodScore = 5.0
        newFoodDistanceScore = 5.0 / (sum(newFoodDistances) + 1)
        newGhostDistances = map(lambda x: manhattan_distance(x.getPosition(), newPos), newGhostStates)
        newGhostDistanceScore = sum(newGhostDistances)/10.0
        # print newFoodDistanceScore, 'food distance score'
        # print newGhostDistanceScore, 'ghost distance score'
        # print foodEatingIncentiveScore, 'food eating incentive score'

        "*** YOUR CODE HERE ***"
        if min(newGhostDistances) > 2 and action == Directions.STOP:
            return -1
        if min(newGhostDistances) < 2:
            return -1
        return newFoodDistanceScore + newGhostDistanceScore + foodEatingIncentiveScore + moveTowardFoodScore
        # return successorGameState.getScore()

def manhattan_distance(s1, s2):
    return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # return Directions.WEST
        min_max_state = self.min_max(gameState, 0, 0)
        print min_max_state
        return min_max_state[0]
        util.raiseNotDefined()

    def min_max(self, gameState, depth, agentIndex, sentAct = None):
        if depth == self.depth:
            return (sentAct, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxScore(gameState, depth, 0)
        else:
            return self.minScore(gameState, depth, agentIndex)

    def minScore(self, gameState, depth, agentIndex):
        min_value = 1e9
        min_direction = None
        max_agent_index = gameState.getNumAgents() - 1
        if max_agent_index == agentIndex:
            new_depth = depth + 1
            new_agent_index = 0
        else:
            new_depth = depth
            new_agent_index = agentIndex + 1
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, new_depth, new_agent_index, act)[1]
            if score < min_value:
                min_value = score
                min_direction = act
        return (min_direction, min_value)

    def maxScore(self, gameState, depth, agentIndex):
        max_value = -1 * 1e9
        max_direction = None
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, depth, agentIndex + 1)[1]
            if score > max_value:
                max_value = score
                max_direction =  act
        return (max_direction, max_value)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        min_max_state = self.min_max(gameState, 0, 0, -1e9, 1e9)
        print min_max_state
        return min_max_state[0]
        util.raiseNotDefined()

    def min_max(self, gameState, depth, agentIndex, alpha, beta, sentAct = None):
        if depth == self.depth:
            return (sentAct, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxScore(gameState, depth, 0, alpha, beta)
        else:
            return self.minScore(gameState, depth, agentIndex, alpha, beta)

    def minScore(self, gameState, depth, agentIndex, alpha, beta):
        min_value = 1e9
        min_direction = None
        max_agent_index = gameState.getNumAgents() - 1
        if max_agent_index == agentIndex:
            new_depth = depth + 1
            new_agent_index = 0
        else:
            new_depth = depth
            new_agent_index = agentIndex + 1
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, new_depth, new_agent_index, alpha, beta, act)[1]
            if score < min_value:
                min_value = score
                min_direction = act
            if min_value < alpha:
                return (min_direction, min_value)
            beta = min(beta, min_value)
        return (min_direction, min_value)

    def maxScore(self, gameState, depth, agentIndex, alpha, beta):
        max_value = -1 * 1e9
        max_direction = None
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, depth, agentIndex + 1, alpha, beta)[1]
            if score > max_value:
                max_value = score
                max_direction =  act
            if max_value > beta:
                return (max_direction, max_value)
            alpha = max(alpha, max_value)
        return (max_direction, max_value)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        min_max_state = self.min_max(gameState, 0, 0)
        print min_max_state
        return min_max_state[0]

    def min_max(self, gameState, depth, agentIndex, sentAct = None):
        if depth == self.depth:
            return (sentAct, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxScore(gameState, depth, 0)
        else:
            return self.minScore(gameState, depth, agentIndex)

    def minScore(self, gameState, depth, agentIndex):
        min_value = 1e9
        min_direction = None
        max_agent_index = gameState.getNumAgents() - 1
        if max_agent_index == agentIndex:
            new_depth = depth + 1
            new_agent_index = 0
        else:
            new_depth = depth
            new_agent_index = agentIndex + 1
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        evaluations = []
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, new_depth, new_agent_index, act)[1]
            evaluations.append(score)
        final_score = sum(evaluations) * 1.0/len(evaluations)
        return (Directions.STOP, final_score)

    def maxScore(self, gameState, depth, agentIndex):
        max_value = -1 * 1e9
        max_direction = None
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions) == 0:
            return (Directions.STOP, self.evaluationFunction(gameState))
        for act in legal_actions:
            succ = gameState.generateSuccessor(agentIndex, act)
            score = self.min_max(succ, depth, agentIndex + 1)[1]
            if score > max_value:
                max_value = score
                max_direction =  act
        return (max_direction, max_value)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newFoodDistances = map(lambda x: manhattan_distance(x, newPos), newFood.asList())
    newFoodDistanceScore = 5.0 / (sum(newFoodDistances) + 1)
    newGhostDistances = map(lambda x: manhattan_distance(x.getPosition(), newPos), newGhostStates)
    newGhostDistanceScore = sum(newGhostDistances)/10.0
    # print newFoodDistanceScore, 'food distance score'
    # print newGhostDistanceScore, 'ghost distance score'
    # print foodEatingIncentiveScore, 'food eating incentive score'

    "*** YOUR CODE HERE ***"
    if min(newGhostDistances) < 2:
        return -1
    return newFoodDistanceScore + newGhostDistanceScore
    # return successorGameState.getScore()


# Abbreviation
better = betterEvaluationFunction

