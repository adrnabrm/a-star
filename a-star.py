"""
Adrian Abraham
Moon
CECS 451
9/13/24

This program will take a map of cities and its coordinates and find the optimal path and its length between a starting and finishing city using the A* algorithm.
"""

import sys
import re
import heapq
import numpy as np

class City:
    """
    City represents a node in the map

    Attributes:
        name (string): the name of the city
        coords (tuple(int, int)): latitude and longitude coords of the city
        _adjacent_cities (dictionary): stores the adjacent cities as keys and the distance between itself and the neighbors as a float as the values

    Methods:
        adjacent_cities(): getter, returns adjacent cities
        adjacent_cities(cities): setter, sets the cities parameter to the adjacent cities variable
        __str__(): overrides string operation to print out name and coords
    """

    def __init__(self, name, x, y):
        self.name = name
        self.coords = (x,y)
        self._adjacent_cities = {}

    @property
    def adjacent_cities(self):
        return self._adjacent_cities

    @adjacent_cities.setter
    def adjacent_cities(self, cities):
        self._adjacent_cities = cities

    def __str__(self):
        return f"{self.name}: {self.coords}"

    def __repr__(self):
        return self.__str__()

def retrieve_cities(coordinates_file, map_file):
    """
    Processes the coordinates and map file
    :param coordinates_file: str
        Path to a file containing coordinates of each city.
    :param map_file:
        Path to a file containing each cities' neighboring cities
    :return: dictionary
        Returns a dictionary where the name of the city is the key and the value is the actual city object
    """

    cities = {}

    # opening coordinates_file
    with open(coordinates_file) as coords:
        # pattern will be: CityName:(latitude,longitude) where latitude and longitude are floats
        coord_pattern = r'(\w+):\(([-+]?\d*\.\d+|\d+),([-+]?\d*\.\d+|\d+)\)'
        for line in coords:
            # using regex (regular expression) to search for the pattern in the line
            match = re.search(coord_pattern, line)
            if match:
                # when found, we can extract the city name and coordinate information and create a city object.
                city, latitude, longitude = match.groups()
                # the object will be stored in a dictionary where the city name is the key
                cities[city] = City(city, float(latitude), float(longitude))

    # opening map_file
    with open(map_file) as map:
        for line in map:
            # extracting the city name
            name = line.split('-')[0]
            # extracting the list of adjacent cities
            adjacent_cities = line.split('-')[1].strip().split(',')
            # the adjacent cities will have a pattern of CityName(distance from root city) where the distance is some float or int value
            adjacent_node_pattern = r'(\w+)\((\d*\.\d+|\d+)\)'
            for adjacent in adjacent_cities:
                # using regex to find the pattern in the file
                match = re.search(adjacent_node_pattern, adjacent)
                if match:
                    # now we set the adjacent_cities found in the map file to its actual City object
                    adjacent_city_name, distance = match.groups()
                    cities[name].adjacent_cities[adjacent_city_name] = float(distance)

    return cities

def straight_line_distance(start, finish):
    """
    Calculated the distance between the starting and finishing city using the Haversine formula

    :param start: City
        The starting point
    :param finish:
        The ending point
    :return:
        The straight line distance between them using the Haversine formula
    """

    r = 3958.8
    # converting each coordinate value to radians for calculation
    phi_1 = np.radians(start.coords[0])
    phi_2 = np.radians(finish.coords[0])
    lambda_1 = np.radians(start.coords[1])
    lambda_2 = np.radians(finish.coords[1])

    h = 2 * r * np.arcsin(np.sqrt( (np.sin((phi_2 - phi_1)/2) ** 2) + np.cos(phi_1) * np.cos(phi_2) * (np.sin((lambda_2 - lambda_1)/2) ** 2) ))

    return h

def a_star(cities, start, finish):
    """
    Implementation of the A* search algorithm which finds the optimal path between the starting and finishing city

    :param cities:
        A dictionary of cities where its name points to its corresponding City object
    :param start:
        Represents the starting city as a City object
    :param finish:
        Represents the ending city as a City object
    :return:
        The optimal path between the start and finish
    """

    # came_from is a dict storing which city the key city was reached from
    came_from = {start: None}
    pq = []
    # cost_to_current contains the g(n) values of each city, meaning how far the city is from the starting point
    cost_to_current = {start: 0}

    heapq.heappush(pq, (0,start))
    while pq:
        # get min from priority queue
        current_cost, current_city = heapq.heappop(pq)
        # if we reach it we're done
        if current_city.name == finish.name:
            path = []
            # using came_from to construct the optimal path
            while current_city:
                path.append(current_city)
                current_city = came_from[current_city]
            return path[::-1]
        # get its adjacent cities
        adjacent_cities = current_city.adjacent_cities
        for neighbor_name, distance in adjacent_cities.items():
            neighbor = cities[neighbor_name]

            # finding the actual cost (g(n)) to reach the current city
            new_cost = cost_to_current[current_city] + distance

            # if the neighbor hasnt been visited or the neighbor now has a lower cost
            if neighbor not in cost_to_current or new_cost < cost_to_current[neighbor]:
                # updating g(n) of the neighbor
                cost_to_current[neighbor] = new_cost
                # adding g(n) to the straight line distance ((h(n)) to get estimated total cost of path (f(n))
                total_cost = new_cost + straight_line_distance(neighbor, finish)
                # we add the neighbor to the heap with its total_cost
                heapq.heappush(pq, (total_cost, neighbor))
                # storing it so that the neighbor city comes from the current city
                came_from[neighbor] = current_city

def calc_path_distance(path):
    """
    Takes a the path given and calculates the distance between the start and end of it
    :param path:
    :return:
    """
    total_distance = 0
    for i in range(len(path)- 1):
        current_city = path[i]
        next_city = path[i + 1]
        total_distance += current_city.adjacent_cities[next_city.name]
    return total_distance

def main():
    """
    Acts as the starting point of the file
    """
    if len(sys.argv) != 3:
        raise ValueError("Expecting 2 cities as input.")

    coordinates_file = 'coordinates.txt'
    map_file = 'map.txt'
    cities = retrieve_cities(coordinates_file, map_file)

    if sys.argv[1] in cities and sys.argv[2] in cities:
        current_location = cities[sys.argv[1]]
        destination = cities[sys.argv[2]]
    else:
        raise ValueError("Starting and/or destination city(s) not found.")

    path = a_star(cities, current_location, destination)
    distance = calc_path_distance(path)

    print(f'From city: {current_location.name}')
    print(f'To city: {destination.name}')
    print(f'Best Route: {' - '.join(city.name for city in path)}')
    print(f'Total distance: {'{0:.2f}'.format(distance)} mi')


if __name__ == "__main__":
    main()