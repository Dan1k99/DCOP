import random
import matplotlib.pyplot as plt
import numpy as np
import copy


class Agent:
    def __init__(self, agent_id, domain_size, neighbor_ids):
        self.id = agent_id
        self.domain_size = domain_size
        self.neighbor_ids = neighbor_ids
        self.current_value = None  # Will be set in setup_problem
        self.received_messages = []
        self.cost_tables = {}
        self.mailbox = []

    # Set up the initial problem state for the agent
    def setup_problem(self, random_state):
        self.current_value = random_state.randint(0, self.domain_size - 1)
        for neighbor_id in self.neighbor_ids:
            self.cost_tables[neighbor_id] = {}
            for my_value in range(self.domain_size):
                for neighbor_value in range(self.domain_size):
                    self.cost_tables[neighbor_id][(my_value, neighbor_value)] = random_state.randint(0, 99)

    # Calculate the total cost based on the agent's current value and its neighbors' values
    def calculate_total_cost(self):
        total_cost = 0
        if self.received_messages:
            latest_neighbor_values = self.received_messages[-1]
            for neighbor_id in self.cost_tables:
                if neighbor_id in latest_neighbor_values:
                    neighbor_value = latest_neighbor_values[neighbor_id]
                    total_cost += self.cost_tables[neighbor_id][(self.current_value, neighbor_value)]
        return total_cost

    # Decide whether to change the current value using the DSA algorithm
    def decide_new_value_dsa(self, change_probability):
        actual_current_value = self.current_value
        current_cost = self.calculate_total_cost()
        best_value = self.current_value
        lowest_cost = current_cost

        for potential_value in range(self.domain_size):
            if potential_value != self.current_value:
                self.current_value = potential_value
                new_cost = self.calculate_total_cost()
                if new_cost < lowest_cost:
                    best_value = potential_value
                    lowest_cost = new_cost

        if lowest_cost < current_cost and random.random() < change_probability:
            self.current_value = best_value
        else:
            self.current_value = actual_current_value

        return {self.id: self.current_value}

    # Decide the new value using the MGM algorithm
    def decide_new_value_mgm(self):
        current_cost = self.calculate_total_cost()
        best_value = self.current_value
        max_gain = 0

        for value in range(self.domain_size):
            if value != self.current_value:
                self.current_value = value
                new_cost = self.calculate_total_cost()
                gain = current_cost - new_cost
                if gain > max_gain:
                    max_gain = gain
                    best_value = value

        self.current_value = best_value
        return best_value, max_gain

    # Send gain messages to neighbors for the MGM algorithm
    def send_gain_message(self):
        _, gain = self.decide_new_value_mgm()
        messages = []
        for neighbor in self.neighbor_ids:
            messages.append((neighbor, (self.id, gain)))
        return messages

    # Process received messages for the MGM algorithm
    def process_messages_mgm(self):
        if not self.mailbox:  # If mailbox is empty
            return  # Do nothing and return

        max_neighbor_gain = max([gain for _, gain in self.mailbox])
        _, my_gain = self.decide_new_value_mgm()

        if my_gain > max_neighbor_gain:
            self.current_value, _ = self.decide_new_value_mgm()

    # Clear all messages from the agent's mailbox
    def clear_mailbox(self):
        self.mailbox = []

    # Store a received message in the agent's message history
    def store_received_message(self, message):
        self.received_messages.append(message)


class DistributedProblemSolver:
    def __init__(self, agent_count, domain_size, neighbor_probability, random_state):
        self.agents = []
        self.setup_agents(agent_count, domain_size, neighbor_probability, random_state)

    # Set up agents with their neighbors and initial problems
    def setup_agents(self, agent_count, domain_size, neighbor_probability, random_state):
        # Create agents
        for i in range(agent_count):
            neighbor_ids = []
            for j in range(agent_count):
                if j != i and random_state.random() < neighbor_probability:
                    neighbor_ids.append(j)
            new_agent = Agent(i, domain_size, neighbor_ids)
            self.agents.append(new_agent)

        # Setup problem for each agent
        for agent in self.agents:
            agent.setup_problem(random_state)

    # Initialize the first round of messages for all agents
    def initialize_received_messages(self):
        for agent in self.agents:
            neighbor_messages = {}
            for neighbor in agent.neighbor_ids:
                neighbor_messages[neighbor] = self.agents[neighbor].current_value
            agent.store_received_message(neighbor_messages)

    # Perform a single iteration of the DSA algorithm
    def perform_single_iteration_dsa(self, change_probability):
        new_values = {}
        for agent in self.agents:
            agent_new_value = agent.decide_new_value_dsa(change_probability)
            new_values.update(agent_new_value)

        for agent in self.agents:
            neighbor_messages = {}
            for neighbor_id in agent.neighbor_ids:
                if neighbor_id in new_values:
                    neighbor_messages[neighbor_id] = new_values[neighbor_id]
            agent.store_received_message(neighbor_messages)

    # Perform a single iteration of the MGM algorithm
    def perform_single_iteration_mgm(self):
        all_messages = []
        for agent in self.agents:
            all_messages.extend(agent.send_gain_message())

        # Then, deliver all messages
        for recipient, message in all_messages:
            self.agents[recipient].mailbox.append(message)

        # Finally, all agents process their messages
        for agent in self.agents:
            agent.process_messages_mgm()
            agent.clear_mailbox()

        # Update received messages for the next iteration
        new_values = {agent.id: agent.current_value for agent in self.agents}
        for agent in self.agents:
            neighbor_messages = {neighbor_id: new_values[neighbor_id] for neighbor_id in agent.neighbor_ids}
            agent.store_received_message(neighbor_messages)

    # Calculate the total cost of the current solution across all agents
    def calculate_global_cost(self):
        total_cost = 0
        for agent in self.agents:
            total_cost += agent.calculate_total_cost()
        return total_cost / 2


# Run a single instance of the specified algorithm
def run_algorithm(agent_count, domain_size, neighbor_probability, iteration_count, algorithm, change_probability,
                  random_state):
    solver = DistributedProblemSolver(agent_count, domain_size, neighbor_probability, random_state)
    solver.initialize_received_messages()  # Initialize messages for all algorithms
    cost_history = []

    for _ in range(iteration_count):
        if algorithm == 'dsa':
            solver.perform_single_iteration_dsa(change_probability)
        elif algorithm == 'mgm':
            solver.perform_single_iteration_mgm()

        current_global_cost = solver.calculate_global_cost()
        cost_history.append(current_global_cost)

    return cost_history


# Run multiple experiments with different parameters and algorithms
def run_experiments(agent_count, domain_size, neighbor_probabilities, iteration_count, num_runs):
    algorithms = [
        ('DSA-C (p=0.2)', 'dsa', 0.2),
        ('DSA-C (p=0.7)', 'dsa', 0.7),
        ('DSA-C (p=1.0)', 'dsa', 1.0),
        ('MGM', 'mgm', None)
    ]

    results = {}

    for k in neighbor_probabilities:
        results[k] = {}
        for alg_name, _, _ in algorithms:
            results[k][alg_name] = []

        for run in range(num_runs):
            random_state = random.Random(123 + run)

            run_results = {alg_name: [] for alg_name, _, _ in algorithms}

            for alg_name, alg_type, change_prob in algorithms:
                cost_history = run_algorithm(agent_count, domain_size, k, iteration_count, alg_type, change_prob,
                                             random_state)
                run_results[alg_name] = cost_history

            # Accumulate results for this run
            for alg_name in run_results:
                if len(results[k][alg_name]) < iteration_count:
                    results[k][alg_name] = run_results[alg_name]
                else:
                    results[k][alg_name] = [results[k][alg_name][i] + run_results[alg_name][i] for i in
                                            range(iteration_count)]

        # Calculate average costs across all runs
        for alg_name in results[k]:
            results[k][alg_name] = [cost / num_runs for cost in results[k][alg_name]]

    return results


# Plot the results of the experiments
def plot_results(results, neighbor_probabilities):
    for k in neighbor_probabilities:
        plt.figure(figsize=(10, 6))
        for alg_name in results[k]:
            costs = results[k][alg_name]
            plt.plot(range(1, len(costs) + 1), costs, label=alg_name)

        plt.xlabel('Iteration')
        plt.ylabel('Average Global Cost')
        plt.title(f'Algorithm Comparison (k={k})')
        plt.legend()
        plt.grid(True)

        # Save the plot
        filename = f'algorithm_comparison_k{k}.png'
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

        # Display the plot
        plt.show()

        plt.close()


if __name__ == "__main__":
    # Set up experiment parameters
    agent_count = 30
    domain_size = 10
    neighbor_probabilities = [0.2, 0.7]
    iteration_count = 100
    num_runs = 30

    # Run experiments and plot results
    results = run_experiments(agent_count, domain_size, neighbor_probabilities, iteration_count, num_runs)
    plot_results(results, neighbor_probabilities)

    print("Execution completed. Check for saved plot files in the current directory.")