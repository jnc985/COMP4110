import os
import numpy as np

from SCP.OR_SCP import Reader


def generate_solution_greedy(matrix):
    # Define univers
    U = list(range(1, matrix.shape[0]))
    # print(U)
    solution = np.zeros(matrix.shape[1])
    #print(solution)

    while len(U) > 0:
        cheapest_column_index = 0
        cheapest_column_cover = []
        cheapest_column_cost = np.inf

        for i, column in enumerate(matrix.T):
            cost = column[0]
            new_cover = [x for x in np.delete(np.where(column)[0], 0) if x in U]

            length = len(new_cover)
            if length > 0:
                value = cost / length
                if value < cheapest_column_cost:
                    cheapest_column_cost = value
                    cheapest_column_cover = new_cover
                    cheapest_column_index = i

        solution[cheapest_column_index] = 1
        U =[u for u in U if u not in cheapest_column_cover]

    print(solution)
    print("Cost", solution_cost(matrix, solution))
    return solution

def is_viable(instance, solution):
    # Create a universe of all the rows
    # universe = list(range(1, self.n + 1))
    universe = list(range(1, instance.shape[0]))
    for col_in_solution in np.where(solution)[0]:
        # Skip the costs, so start enumeration at 1 aswell
        for j, col_in_matrix in enumerate(instance[1:, col_in_solution], 1):
            if col_in_matrix and j in universe:
                universe.remove(j)
                if len(universe) == 0:
                    return True, universe
    return False, universe

def find_neighbour(matrix, solution):
    fail_count = 0
    viable = False
    while not viable and fail_count < matrix.shape[1]:
        neighbour = solution.copy()
        index = np.random.randint(0, matrix.shape[0])
        neighbour[index] = not neighbour[index]
        viable = is_viable(matrix, neighbour)[0]
        if not viable:
            fail_count += 1
    return neighbour


def simmulated_annealing(matrix, solution):
    T = 1

    best_cost = current_cost = solution_cost(matrix, solution)
    best_solution = current_solution = solution

    cooling_rate = 0.92

    min_temp = 0.001

    while T > min_temp:
        # Pick a random neightbour
        neighbour = find_neighbour(matrix, current_solution)
        neighbour_cost = solution_cost(matrix, neighbour)
        # Should we move to this neighbour?
        probability_to_accept = accept_probability(current_cost, neighbour_cost, T)
        if probability_to_accept >= np.random.random():
            print("BETTER: ", is_viable(matrix, neighbour))
            current_solution = neighbour
            current_cost = neighbour_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = current_solution
            print("New Best: ", current_cost)

        # Update/Cool the system
        T = T * cooling_rate


    print(best_solution)
    print(is_viable(matrix, best_solution))
    print(best_cost)

def accept_probability(current_cost, neighbour_cost, temperarture):
    if neighbour_cost < current_cost:
        return 1
    else:
        prob = np.exp(-(neighbour_cost - current_cost) / temperarture)
        return prob

def solution_cost(instance, solution, length=None):
    if length == None:
        length = len(solution)
    if instance.ndim == 2:
        return sum(instance[0][i] for i in np.where(solution[:length])[0])
    else:
        return sum(instance[i] for i in np.where(solution[:length])[0])



if __name__ == '__main__':
    directory = os.path.dirname(__file__)
    dataDirectory = os.path.join(directory, '../OR_Data/Instances')

    scpInstance = Reader.read_file(os.path.join(dataDirectory, 'scp41.txt'))
    #print(scpInstance)
    solution = generate_solution_greedy(scpInstance)
    simmulated_annealing(scpInstance, solution)