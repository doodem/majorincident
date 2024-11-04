import networkx as nx
import numpy as np
import math
import random
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ipywidgets as widgets
from IPython.display import display, clear_output

def VoronoiGraphRoadNetwork(world, seed=None):
    """Initiates process to build a networkx graph with certain attributes.

    The world parameter in MajorCrimeIncidentResponse() accepts a networkx
    graph where edges have a 'dist' attribute representing the distance of roads 
    and a 'station' attributre indicating which nodes are police stations.  
    Agent responders start from these station nodes. The 'pos' attribute must
    also be set storing the position of nodes.
    """
    if seed is not None:
        np.random.seed(seed)
    
    points = add_clusters_to_points(world)
    G, pos = voronoi_to_networkx(points)
    
    for u, v in G.edges():
        G.edges[u, v]['dist'] = distance(u, v, pos)
    
    nx.set_node_attributes(G, pos, 'pos')
    G = set_node_stations(G, world["station_positions"])
    
    return G

def add_clusters_to_points(world):
    """Adds non-uniformity to points. 

    Non-uniformity in the graph shows as denser areas of edges.
    This better emulates real-world road networks.
    """
    num_clusters = world["cluster_number"]
    cluster_size = world["cluster_size"]
    cluster_points = world["cluster_points"]
    road_network_points = world["road_network_points"]

    points = np.random.rand(road_network_points, 2)
    
    cluster_means = np.random.rand(num_clusters, 2)
    cluster_stds = np.random.rand(num_clusters) * cluster_size

    for mean, std in zip(cluster_means, cluster_stds):
        cluster = np.random.normal(loc=mean, scale=std, size=(cluster_points, 2))
        points = np.concatenate([points, cluster])

    return points

def voronoi_to_networkx(points):
    """Builds a networkx voronoi graph given a number of points. 

    Voronoi graphs are planar graphs, meaning that there are no
    overlapping edges. These graphs are often used to model geometry.
    """
    vor = Voronoi(points)
    G = nx.Graph()
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            i, j = simplex
            p, q = vor.vertices[i], vor.vertices[j]
            if all(0 <= x <= 1 for x in p) and all(0 <= x <= 1 for x in q):
                distance = np.linalg.norm(p - q)
                G.add_edge(tuple(p), tuple(q), weight=distance)
    pos = {i: node for i, node in enumerate(G.nodes)}
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    return G, pos

def distance(u, v, pos):
    """Returns Euclidean distance betwween two nodes.
    """
    return math.ceil(math.dist(pos[u], pos[v]) * 100)

def set_node_stations(graph, coords_list):
    """Sets the closest node to provided coordinates with a 'station' attribute.
    """
    pos = nx.get_node_attributes(graph, 'pos')
    for coords in coords_list:
        station_node = min(graph.nodes(), key=lambda node: math.dist(pos[node], coords))
        graph.nodes[station_node]['station'] = True
    return graph

def PlotResponse(ax, data, x, y, sd=None, group=None, label=None, colour=['#666666', '#FDBF2D', '#4C67DD', '#F56665', '#BEBEBE']):
    """Plots results of response.
    
    example:
    >>> data = pd.read_csv('data.csv')
    >>> fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    >>> PlotResponse(axes[0], data4, 'TimeStep', 'Avg_Gathered', 'SD_Gathered', 'ModelNumber')
    >>> PlotResponse(axes[1], data4, 'TimeStep', 'Avg_Reached', 'SD_Reached', 'ModelNumber')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    label_map = {
        'Pheromone': r'$p$',
        'TimeStep': r'$t$',
        'Avg_Gathered': r'$G(t)$',  
        'Avg_Time': r'$t$',  
        'Avg_Covered': r'$C(t)$',   
        'Avg_Reached': r'$R(t)$',
        'Avg_Hits': r'$\Delta C(t)$', 
        'Avg_Equality': r'$U(t)$'      
    }

    ax.set_prop_cycle('color', colour)

    grouped = data.groupby(group)

    for _, group in grouped:
        if label is not None:
            ax.plot(group[x], group[y], linewidth=3, label=group[label].iloc[0])
        else:
            ax.plot(group[x], group[y], linewidth=3)

        if sd is not None:
            ax.fill_between(group[x], 
                            group[y] - group[sd], 
                            group[y] + group[sd], 
                            alpha=0.5)

    if label is not None:
        ax.legend()

    ax.set_xlabel(label_map[x], fontsize = 22)
    ax.set_ylabel(label_map[y], fontsize = 22)

class AgentResponder:
    
    def __init__(self, world, pheromone_deposit, 
                 emerging_incident, emerging_incident_k, emerging_incident_n, 
                 spreading_incident_ego_graph, spreading_incident_threshold):
        """Initialise agent responder attributes. 
        """
        self.world = world
        self.pos = np.array((0, 0))
        self.current_node = np.random.choice(list(self.world.stations))
        self.coordinates = np.array(((0, 0), (0, 0)))
        self.coordinates_list = np.empty((0, 2))
        self.pheromone_deposit = pheromone_deposit
        self.emerging_incident = emerging_incident
        self.emerging_incident_k = emerging_incident_k
        self.emerging_incident_n = emerging_incident_n
        self.spreading_incident_ego_graph = spreading_incident_ego_graph
        self.spreading_incident_threshold = spreading_incident_threshold
        self.target_incident = self.world.incident
        self.information_gathered = 0
        self.reached_incident = False
        self.active = False

    def move(self): 
        """Controls the movement and decision-making at each node towards the incident.
        """
        if self.at_next_node():
            if self.at_incident():
                self.reached_incident = True
                self.coordinates_list = np.vstack((self.coordinates_list, self.world.pos[self.target_incident]))
            else:
                next_node = self.short_alternative_path()
                if self.current_node != next_node:
                    self.update_pheromones(next_node)
                    self.coordinates = self.steps_to_next_node(next_node)
                    self.current_node = next_node
                    self.pos = self.coordinates[0]
                    self.coordinates_list = np.vstack((self.coordinates_list, self.pos))
                else:
                    self.current_node = next_node
                    self.pos = self.coordinates[-1]
                    self.coordinates_list = np.vstack((self.coordinates_list, self.pos))
        else:
            self.pos = self.coordinates[np.where(self.pos == self.coordinates)[0][0] + 1]
            self.coordinates_list = np.vstack((self.coordinates_list, self.pos))
    
    def at_incident(self):
        """Whether the responder has reached the incident node.
        """
        return self.current_node == self.target_incident
    
    def at_next_node(self):
        """Whether responder has reached the next node.
        """
        return np.array_equal(self.pos, self.coordinates[-1])
    
    def sum_pheromone_ego_grpah(self, target_incident):
        """Sum the pheromones within a radius of an incident.
        """
        subgraph = nx.ego_graph(self.world.graph, target_incident, radius=self.spreading_incident_ego_graph)
        return sum([subgraph[u][v]['cost'] for u, v in subgraph.edges()])
    
    def choose_target_incident(self):
        """Chooses which incident to path towards of mulitple incidents.
        """
        original_incident = self.sum_pheromone_ego_grpah(self.world.incident)
        spreaded_incident = self.sum_pheromone_ego_grpah(self.world.spreaded)
        if abs(original_incident - spreaded_incident) > self.spreading_incident_threshold:
            if original_incident < spreaded_incident:
                self.target_incident = self.world.incident
            else:
                self.target_incident = self.world.spreaded

    def short_alternative_path(self):
        """Returns the next node along the shorest alternative path relative to other responders.
        """
        if self.world.spreaded is not None:
            self.choose_target_incident()
        path = nx.astar_path(self.world.graph, self.current_node, self.target_incident, heuristic=self.euclidian_distance, weight=self.cost_function)
        if len(path) > 1:
            return path[1]
        else:
            return self.current_node
    
    def dynamic_pheromone_deposit(self):
        """Dynamic deposit based on distance (inverse exponential relationship) to incident node.
        
        Smaller deposits farther from the incident (priortise response time) 
        and exponentially larger deposits closer to the incident (priortise information gathering).
        This places more importance on information closer to the incident.
        Parameter `k` ensures inverse relationship and `n` controls magnitude of effect by 
        introducing non-linearity.
        """
        distance = self.euclidian_distance(self.current_node, self.target_incident)
        return self.pheromone_deposit / (distance / self.emerging_incident_k) ** self.emerging_incident_n
    
    def update_pheromones(self, next_node):
        """Calculates the amount of pheromone to be deposited.
        """
        if self.emerging_incident:
            update = self.dynamic_pheromone_deposit()
        else:
            update = self.pheromone_deposit
        self.world.graph.edges[self.current_node, next_node]['cost'] += update
        if self.world.graph.edges[self.current_node, next_node]['info'] == 1:
            self.world.graph.edges[self.current_node, next_node]['info'] = 0
            self.information_gathered += 1

    def cost_function(self, u, v, e):
        """Alters cost of shortest path considering pheromones.

        A proportion of the edge's distance is added to itself
        based on the pheromonmes on such edges. 
        """
        alter = e['dist'] + (e['dist'] * e['cost'])
        return alter

    def euclidian_distance(self, a, b):
        """Returns Euclidian distance bewteen two nodes.
        """
        return math.dist(self.world.pos[a], self.world.pos[b])
    
    def steps_to_next_node(self, next_node):
        """Returns evenly spaced steps between nodes.
        """
        steps = np.linspace(0, 1, self.world.graph[self.current_node][next_node]['dist'])
        return self.positions_to_next_node(steps, next_node)
    
    def positions_to_next_node(self, steps, next_node):
        """Returns positions of steps between nodes.
        """
        return steps[..., None] * self.world.pos[next_node] + (1 - steps[..., None]) * self.world.pos[self.current_node]      

class IncidentScenario:
    def __init__(self, world, search_radius_size, evolving_incident, evolving_incident_growth_by, evolving_incident_growth_delta,
                 evolving_incident_regenerate_delta, spreading_incident, spreading_incident_probability, 
                 spreading_incident_time_multiplier, pheromone_decay):
        """Initialise incident scenario attributes.
        """
        self.graph = world
        self.pos = nx.get_node_attributes(world, 'pos')
        self.stations = [i for i in nx.get_node_attributes(world, 'station').keys()]
        self.incident = random.choice(list([i for i in self.graph.nodes() if i not in self.stations]))
        self.search_radius_size = search_radius_size
        self.search_radius = 0
        self.evolving_incident = evolving_incident
        self.evolving_incident_growth_by = evolving_incident_growth_by
        self.evolving_incident_growth_delta = evolving_incident_growth_delta
        self.evolving_incident_regenerate_delta = evolving_incident_regenerate_delta
        self.spreading_incident = spreading_incident
        self.spreaded = None
        self.spreading_incident_probability = spreading_incident_probability
        self.spreading_incident_time_multiplier = spreading_incident_time_multiplier
        self.pheromone_decay = pheromone_decay
        self.add_search_radius(self.search_radius_size)
        self.edge_map = []
        self.node_map = []
        self.colour_map = []
        self.countdowns = {}
        self.max_pheromone = 0
        self.time = 0    

    def add_search_radius(self, max_dist):
        """Initalise search radius and base graph attributes.
        """
        nx.set_edge_attributes(self.graph, 0, 'cost')
        nx.set_edge_attributes(self.graph, 0, 'info')
        nx.set_edge_attributes(self.graph, 'k', 'colour')
        nx.set_edge_attributes(self.graph, 'N', 'in_search_radius')
        self.grow_search_radius(max_dist)
    
    def grow_search_radius(self, max_dist):
        """Grows the search radius upon initalisation and for evolving incident scenarios.

        This function is also important for keeping a count 
        of the number of edges within the search radius. 
        This is used to measure search radius coverage.
        """
        target = self.incident
        for u, v in self.graph.edges():
            euclidean_dist_to_target = self.distance(u, target, self.pos)
            if euclidean_dist_to_target <= max_dist:
                if self.graph[u][v]['in_search_radius'] != 'Y':
                    self.graph[u][v].update({'colour': 'orange', 'info': 1, 'in_search_radius': 'Y'})
                    self.search_radius += 1
    
    def distance(self, u, v, pos):
        """Calculates Euclidian distance between nodes.
        """
        return math.ceil(math.dist(pos[u], pos[v]) * 100)
    
    def regenerate(self):
        """Regenerates information on edges after a countdown once collected.
        """
        for u, v in self.graph.edges():
            if self.graph[u][v]['in_search_radius'] == "Y" and self.graph[u][v]['info'] == 0:
                if (u, v) not in self.countdowns:
                    self.countdowns[(u, v)] = self.evolving_incident_regenerate_delta
        for (u, v) in list(self.countdowns.keys()):
            if self.countdowns[(u, v)] > 0:
                self.countdowns[(u, v)] -= 1
                if self.countdowns[(u, v)] == 0:
                    self.graph[u][v].update({'colour': 'orange', 'info': 1})
                    del self.countdowns[(u, v)]
                    
    def spread_prob(self):
        """Calculates the probability of an additional incident occuring within the search radius.
        """
        base_probability = self.spreading_incident_probability
        time_multiplier = self.spreading_incident_time_multiplier
        probability = base_probability + time_multiplier * self.time
        probability = min(probability, 1.0)
        return probability

    def check_spread(self):
        """Places the additonal incident within the search radius.
        """
        probability = self.spread_prob()
        if random.random() < probability and self.spreaded is None:
            search_radius_nodes = set()
            for u, v in self.graph.edges():
                if self.graph[u][v]['in_search_radius'] == "Y":
                    search_radius_nodes.add(u)
                    search_radius_nodes.add(v)
            valid_spread_nodes = [node for node in search_radius_nodes if node not in self.stations and node != self.incident]
            if valid_spread_nodes:
                self.spreaded = np.random.choice(valid_spread_nodes)

    def gen_maps(self):
        """Generates maps for the simulation to visualise different characteristics of the incident at time t.
        """
        colour_map = np.full(len(self.graph.nodes()), "g")
        colour_map[self.incident] = "r"
        colour_map[self.stations] = "c"
        node_map = np.zeros(len(self.graph.nodes()))
        node_map[self.incident] = 150
        node_map[self.stations] = 150
        if self.spreaded is not None:
            colour_map[self.spreaded] = "r"
            node_map[self.spreaded] = 100
        self.node_map.append(node_map)
        self.colour_map.append(colour_map)
        self.edge_map.append([self.graph[u][v]['colour'] for u, v in self.graph.edges()])

    def update(self):
        """Updates the chosen incident scenario.
        """
        if self.spreading_incident:
            self.check_spread()
            
        if self.evolving_incident:
            self.regenerate()
            if self.evolving_incident_growth_delta > 0:
                if self.time % self.evolving_incident_growth_delta == 0:
                    self.search_radius_size += self.evolving_incident_growth_by
                    self.grow_search_radius(self.search_radius_size)

        for u, v in self.graph.edges():
            self.graph[u][v]['cost'] *= (1 - self.pheromone_decay)
            new_pheromone = self.graph[u][v]['cost']
            info_exists = self.graph[u][v]['info']
            if new_pheromone > self.max_pheromone:
                self.max_pheromone = new_pheromone
            if new_pheromone > 0 and self.graph[u][v]['info'] != 1:
                alpha = min(1, new_pheromone / self.max_pheromone)
                alpha = max(alpha, 0.1)
                self.graph[u][v]['colour'] = (0.6, 0.0, 1.0, alpha)
            else:
                if info_exists:
                    self.graph[u][v]['colour'] = 'orange'
                else:
                    self.graph[u][v]['colour'] = 'k'
        self.gen_maps()

class MajorCrimeIncidentResponse:
    
    def __init__(self, world, search_radius_size=25, number_of_responders=30, pheromone_deposit=0.5, pheromone_decay=0,
                 staggered_dispatch=False, staggered_dispatch_responders=10, staggered_dispatch_delta=20,
                 evolving_incident=False, evolving_incident_growth_by=0, evolving_incident_growth_delta=0, evolving_incident_regenerate_delta=0,
                 emerging_incident=False, emerging_incident_k=0.2, emerging_incident_n=1.5,
                 spreading_incident=False, spreading_incident_ego_graph=1, spreading_incident_threshold=0.5, spreading_incident_probability=0.05, spreading_incident_time_multiplier=0.001):
        """Initalise simulation parameters. 
        """
        self.world = IncidentScenario(world, search_radius_size, 
                           evolving_incident, evolving_incident_growth_by, evolving_incident_growth_delta, evolving_incident_regenerate_delta, 
                           spreading_incident, spreading_incident_probability, spreading_incident_time_multiplier,
                           pheromone_decay)
        self.responders = [AgentResponder(self.world, pheromone_deposit, 
                                          emerging_incident, emerging_incident_k, emerging_incident_n,
                                          spreading_incident_ego_graph, spreading_incident_threshold) for _ in range(number_of_responders)]
        self.instruments = []
        self.staggered_dispatch = staggered_dispatch
        self.staggered_dispatch_delta = staggered_dispatch_delta
        self.staggered_dispatch_responders = staggered_dispatch_responders
        self.instruments = []
        self.current_frame = 0
        self.frame_label = widgets.Label(value=f"Frame: {self.current_frame}")

    def run(self):
        """Run simulation for a given scenario.
        """
        self.update_instruments()
        while not all(responder.reached_incident for responder in self.responders):             
                self.step()
                self.world.time += 1
            
    def step(self):
        """Steps world instance and each agent through simulation.
        """
        if self.staggered_dispatch:
            self.dispatch()
        for responder in self.responders:
            if responder.active == self.staggered_dispatch:
                responder.move()
        self.update_instruments()
        self.world.update()
        
    def dispatch(self):
        """Staggers dispatch of n responders at every chosen interval.

        Will dispatch rest of responders at the next interval if 
        number_of_responders not divisible by staggered_dispatch_responders
        """
        if self.world.time % self.staggered_dispatch_delta == 0:
            inactives = [responder for responder in self.responders if not responder.active]
            if inactives:
                inactives_left = min(len(inactives), self.staggered_dispatch_responders)
                for responder in np.random.choice(inactives, inactives_left, replace = False):
                    responder.active = True

    def get_coordinates_list(self):
        """Returns coordinate list of agent movement on graph.
        """
        if self.staggered_dispatch:
            self.pad()       
        coords = np.array([responder.coordinates_list for responder in self.responders])        
        return coords[:, :, 0], coords[:, :, 1]
    
    def pad(self):
        """Pads coordinate list for later dispatched responders.
        """
        max_length = max(len(responder.coordinates_list) for responder in self.responders)
        for responder in self.responders:
            while len(responder.coordinates_list) < max_length:
                responder.coordinates_list = np.concatenate([[(None, None)], responder.coordinates_list], axis=0)       
    
    def add_instrument(self, instrument):
        """Adds chosen metrics to simulation.
        """
        self.instruments.append(instrument)
    
    def update_instruments(self):
        """Updates chosen metrics at time t.
        """
        for instrument in self.instruments:
            instrument.update(self)
    
    def information_gathered(self):
        """Updates information points gathered.
        """
        return sum([responder.information_gathered for responder in self.responders])
    
    def search_radius_covered(self):
        """Updates proportion of search radius covered.
        """
        return self.information_gathered() / self.world.search_radius
    
    def responders_reached_incident(self):
        """Updates number of responders at incident.
        """
        return len([responder for responder in self.responders if responder.reached_incident == True])
    
    def information_equality(self):
        """Updates equality of information acorss all responders.
        """
        return np.std([responder.information_gathered for responder in self.responders])
    
    def graph(self):
        fig = plt.figure()
        return nx.draw(self.world.graph, self.world.pos, node_size=self.world.node_map[1], node_color=self.world.colour_map[1], edge_color=self.world.edge_map[1])
    
    def sim(self, x=None, y=None, zoom=None):
        """Outputs gif of simulation.
        """
        x_coords, y_coords = self.get_coordinates_list()
        fig = plt.figure()

        def animate(i):
            plt.cla()
            nx.draw(self.world.graph, self.world.pos, node_size=self.world.node_map[i], node_color=self.world.colour_map[i], edge_color=self.world.edge_map[i], width = 2)
            plt.plot(x_coords[:, i], y_coords[:, i], 'bs', markersize=5)
            plt.text(0.05, 1, f"t = {i}")

        return animation.FuncAnimation(fig, animate, frames=len(x_coords[0]), interval=1, repeat=False)

    def animate_metrics(self):
        """Outputs gif of results per time step for multiple instruments.
        """
        indices = list(range(len(self.instruments)))

        if len(indices) == 1:
            fig, ax = plt.subplots(figsize=(3, 3))
            axs = [ax]
        else:
            fig, axs = plt.subplots(len(indices), figsize=(3, 3 * len(indices)))

        results = [self.instruments[index].metrics for index in indices]
        names = [self.instruments[index].names for index in indices]
        time_steps = [list(range(1, len(results[i]) + 1)) for i in range(len(indices))]

        def update(i):
            for ax, result, name, steps in zip(axs, results, names, time_steps):
                ax.cla()
                ax.plot(steps[:i], result[:i])
                ax.set_ylabel(name[0], fontsize=5)
                ax.set_xlabel("time steps", fontsize=5)
                ax.tick_params(axis='both', labelsize=5)

        return animation.FuncAnimation(fig, update, frames=100, interval=50)
    
    def frames(self, x=None, y=None, zoom=None):
        """Frame-by-frame of simulation in jupyter notebook.
        """
        x_coords, y_coords = self.get_coordinates_list()
        fig = plt.figure()

        def animate(i, x, y, zoom):
            plt.cla()
            set_zoom(x, y, zoom)
        
            nx.draw(self.world.graph, self.world.pos, node_size=self.world.node_map[i], node_color=self.world.colour_map[i], edge_color=self.world.edge_map[i], width = 2)
            plt.plot(x_coords[:, i], y_coords[:, i], 'bs', markersize=5)
        
        def set_zoom(x, y, zoom):
            if x is not None and y is not None and zoom is not None:
                x_min, x_max = x - zoom/2, x + zoom/2
                y_min, y_max = y - zoom/2, y + zoom/2
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

        def next_frame(b):
            self.current_frame += 1
            if self.current_frame < len(x_coords[0]):
                animate(self.current_frame, x, y, zoom)
                clear_output(wait=True)
                self.frame_label.value = f"Frame: {self.current_frame}"

        def prev_frame(b):
            self.current_frame -= 1
            if self.current_frame >= 0:
                animate(self.current_frame, x, y, zoom)
                clear_output(wait=True)
                self.frame_label.value = f"Frame: {self.current_frame}"

        button_next = widgets.Button(description="Next Frame")
        button_next.on_click(next_frame)

        button_prev = widgets.Button(description="Previous Frame")
        button_prev.on_click(prev_frame)

        display(widgets.HBox([button_prev, button_next]), fig, self.frame_label)
        animate(self.current_frame, x, y, zoom)

class Instrument:
        
    def __init__(self):
        """ Stores chosen metrics.
        """
        self.metrics = []
        self.names = []

class InformationGathering(Instrument):
    
    def __init__(self):
        """Collects information gathered.
    
        A major crime incident response must gather information. More information gathering 
        may resolve the incident faster, rather than soley relying on a fast response. 
        This is measured by the total number of information points gathered by responders for time t. 
        """
        super().__init__()
        self.names.append('Information Gathering')
    
    def update(self, model):
        gathered = model.information_gathered()
        self.metrics.append(gathered)

class SearchRadiusCoverage(Instrument):
    
    def __init__(self):
        """Collects proportion of search radius covered.

        Not all edges hold information points. Only edges within a search radius around the
        incident. This is measured as the proportion of information points gathered
        within the search radius for time t. Will adjust based on an expanding raduis
        or regenerating information.
        """
        super().__init__()
        self.names.append('Search Radius Coverage')
    
    def update(self, model):
        covered = model.search_radius_covered()
        self.metrics.append(covered)

class ResponderReachedIncident(Instrument):
   
    def __init__(self):
        """Collects number of responders at incident node(s).
    
        A major crime incident response must be fast. 
        This is measured by the number of responders at incident node(s) for time t.
        """
        super().__init__()
        self.names.append('Responders Reached Incident')

    def update(self, model):
        reached = model.responders_reached_incident()
        self.metrics.append(reached)

class InformationEquality(Instrument):
    
    def __init__(self):
        """Collect the distribution of information gathered across responders.

        Ideally, each responder would have an equal amount of information
        gathererd representing full utilisation of all responders.
        """
        super().__init__()
        self.names.append('Information Equality')

    def update(self, model):
        dist = model.information_equality()
        self.metrics.append(dist)