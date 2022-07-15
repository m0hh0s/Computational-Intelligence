import copy
import random
from player import Player
import numpy as np
from config import CONFIG


class Evolution:
	def __init__(self, mode):
		self.mode = mode

	def calculate_fitness(self, players, delta_xs):
		for i, p in enumerate(players):
			p.fitness = delta_xs[i]

	def mutate(self, children, mutation_probability=0.3):
		mu = 0
		sigma = 0.5
		for child in children:
			for weight1 in child.nn.w1:
				if random.random() < mutation_probability:
					weight1 += np.random.normal(mu, sigma)
			for weight2 in child.nn.w2:
				if random.random() < mutation_probability:
					weight2 += np.random.normal(mu, sigma)
			for bias1 in child.nn.b1:
				if random.random() < mutation_probability:
					bias1 += np.random.normal(mu, sigma)
			for bias2 in child.nn.b2:
				if random.random() < mutation_probability:
					bias2 += np.random.normal(mu, sigma)

	def crossover(self, parent1, parent2):
		child = Player(parent1.mode)
		child.nn.w1 = copy.deepcopy(parent1.nn.w1)
		child.nn.w2 = copy.deepcopy(parent2.nn.w2)
		child.nn.b1 = copy.deepcopy(parent1.nn.b1)
		child.nn.b2 = copy.deepcopy(parent2.nn.b2)
		return child

	def generate_new_population(self, num_players, prev_players=None):
		if prev_players is None:
			return [Player(self.mode) for _ in range(num_players)]
		else:
			fitness_weights = [prev_players[j].fitness ** 4 for j in range(num_players)]
			new_players = []
			prev_players = copy.deepcopy(prev_players)
			self.mutate(prev_players)
			for _ in range(num_players):
				p1, p2 = random.choices(prev_players, k=2, weights=fitness_weights)
				child = self.crossover(p1, p2)
				new_players.append(child)
			self.mutate(new_players)
			# TODO (additional): a selection method other than `fitness proportionate`
			# TODO (additional): implementing crossover
			return new_players

	def next_population_selection(self, players, num_players):
		players.sort(key=lambda x: x.fitness, reverse=True)
		# TODO (additional): a selection method other than `top-k`
		# TODO (additional): plotting
		return players[:num_players]
