#!/usr/bin/env python

import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
from majorincident import VoronoiGraphRoadNetwork, MajorCrimeIncidentResponse, InformationGathering, SearchRadiusCoverage, ResponderReachedIncident, InformationEquality

class Ensemble:

    def __init__(self, method, runs, 
                 search_radius_size, number_of_responders, pheromone_deposit, pheromone_decay, 
                 staggered_dispatch, staggered_dispatch_responders, staggered_dispatch_delta,
                 evolving_incident, evolving_incident_growth_by, evolving_incident_growth_delta, evolving_incident_regenerate_delta,
                 emerging_incident, emerging_incident_k, emerging_incident_n,
                 spreading_incident, spreading_incident_ego_graph, spreading_incident_threshold, spreading_incident_probability, spreading_incident_time_multiplier,
                 seed):
        """Initalise simulation parameters.
        """
        self.method = method
        self.runs = runs
        self.search_radius_size = search_radius_size
        self.number_of_responders = number_of_responders
        self.pheromone_deposit = pheromone_deposit
        self.pheromone_decay = pheromone_decay
        self.staggered_dispatch = staggered_dispatch
        self.staggered_dispatch_responders = staggered_dispatch_responders
        self.staggered_dispatch_delta = staggered_dispatch_delta
        self.evolving_incident = evolving_incident
        self.evolving_incident_growth_by = evolving_incident_growth_by
        self.evolving_incident_growth_delta = evolving_incident_growth_delta
        self.evolving_incident_regenerate_delta = evolving_incident_regenerate_delta
        self.emerging_incident = emerging_incident
        self.emerging_incident_k = emerging_incident_k
        self.emerging_incident_n = emerging_incident_n
        self.spreading_incident = spreading_incident
        self.spreading_incident_ego_graph = spreading_incident_ego_graph
        self.spreading_incident_threshold = spreading_incident_threshold
        self.spreading_incident_probability = spreading_incident_probability
        self.spreading_incident_time_multiplier = spreading_incident_time_multiplier
        self.world = VoronoiGraphRoadNetwork(
            world = {
            'road_network_points': 150, 
            'station_positions': [[0.3, 0.7], [0.7,0.7], [0.5, 0.15]],
            'cluster_number': 2,
            'cluster_points': 50,
            'cluster_size': 0.05
        },
        seed = seed)
        self.p = None

    def run(self):
        """Runs ensemble simulations for chosen method.
        """

        if self.method == 1:
            return self.ResponseByTimeStep()
        elif self.method == 2:
            return self.ResponseBalance()
        elif self.method == 3:
            return self.Pareto()
        
    def ResponseByTimeStep(self):
        """Aggregates results per time step.

        This method iterates through a set of pheromone values, runs n simulations,
        and aggregates the results for information gathering, incident coverage, 
        and the number of responders that have reached the incident at each time step.
        """
        
        model_numbers = []
        pheromones = []
        avg_gathered = []
        sd_gathered = []
        avg_covered = []
        sd_covered = []
        avg_reached = []
        sd_reached = []
        avg_hits = []
        sd_hits = []
        avg_equality = []
        sd_equality = []
        time_steps = []

        for i, p in enumerate(self.pheromone_deposit):
            self.p = p

            gathered, covered, reached, equality = self.run_multiple()

            hits = self.convert_non_cum(covered)

            meang, sdg = self.average_across_lengths(gathered)
            meanc, sdc = self.average_across_lengths(covered)
            meanr, sdr = self.average_across_lengths(reached)
            meanh, sdh = self.average_across_lengths(hits)
            meane, sde = self.average_across_lengths(equality)

            num_steps = len(meang)
            model_numbers.extend([i] * num_steps)
            pheromones.extend([p] * num_steps)
            avg_gathered.extend(meang)
            sd_gathered.extend(sdg)
            avg_covered.extend(meanc)
            sd_covered.extend(sdc)
            avg_reached.extend(meanr)
            sd_reached.extend(sdr)
            avg_hits.extend(meanh)
            sd_hits.extend(sdh)
            avg_equality.extend(meane)
            sd_equality.extend(sde)
            time_steps.extend(range(num_steps))

        df = pd.DataFrame({
            'ModelNumber': model_numbers,
            'Pheromone': pheromones,
            'Avg_Gathered': avg_gathered, 
            'SD_Gathered': sd_gathered,
            'Avg_Covered': avg_covered, 
            'SD_Covered': sd_covered,
            'Avg_Reached': avg_reached,
            'SD_Reached': sd_reached,
            'Avg_Hits': avg_hits,
            'SD_Hits': sd_hits,
            'Avg_Equality': avg_equality,
            'SD_Equality': sd_equality,
            'TimeStep': time_steps
        })

        return df

    def ResponseBalance(self):
        """Aggregates final results across simulations.

        This method iterates through a set of pheromone values, runs n simulations,
        and aggregates the final results for proportion of search radius covered and response time.
        """
        
        model_numbers = []
        pheromones = []
        avg_covered = []
        sd_covered = []
        avg_time = []
        sd_time = []

        for i, p in enumerate(self.pheromone_deposit):
            self.p = p

            _, covered, _, _ = self.run_multiple() 
            total = [c[-1] for c in covered]
            time = [len(c) for c in covered]

            model_numbers.append(i)
            pheromones.append(p)
            avg_covered.append(np.mean(total))
            sd_covered.append(np.std(total))
            avg_time.append(np.mean(time))
            sd_time.append(np.std(time))

        df = pd.DataFrame({
            'ModelNumber': model_numbers,
            'Pheromone': pheromones,
            'Avg_Covered': avg_covered,
            'SD_Covered': sd_covered,
            'Avg_Time': avg_time,
            'SD_Time': sd_time
        })

        return df
    
    def Pareto(self):
        """Finds the pareto front based on pheromone and decay combinations.
        
        Calculates the pareto front based on all combinations of pheromone and decay
        with with a 0.1 interval. For example, the find combination is p = 0, d = 0. 
        The second is p = 0.1, d = 0, etc. This method runs n simulations per combination.
        """

        combinations = [(p, d) for p in np.arange(0, 1 + 0.1, 0.1)
                        for d in np.arange(0, 1 + 0.1, 0.1)]

        data = []
        for p, d in combinations:
            self.p = p
            self.decay = d

            _, covered, _, _ = self.run_multiple()

            avg_covered = np.mean([result[-1] for result in covered])
            avg_time = np.mean([len(result) for result in covered])
            data.append((p, d, avg_time, avg_covered))
        
        data.sort(key=lambda point: (point[2], -point[3]))

        pareto_set = []
        for i, point in enumerate(data):
            is_pareto = True
            for other_point in data[:i]:
                if (other_point[2] < point[2] and other_point[3] >= point[3]) or (other_point[2] <= point[2] and other_point[3] > point[3]):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_set.append(point)

        df_data = []
        for point in data:
            status = "Pareto" if point in pareto_set else "Dominated"
            df_data.append((*point, status))

        df = pd.DataFrame(df_data, columns=["Pheromone", "Decay", "Avg_Time", "Avg_Covered", "ParetoFront"])

        return df
    
    def run_multiple(self):
        """Runs simulation n times. 
        """
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.metric, [self.p] * self.runs), 
                                total=self.runs, desc="Running simulations"))
        
        gathered = [result[0] for result in results]
        covered = [result[1] for result in results]
        reached = [result[2] for result in results]
        equality = [result[3] for result in results]
        return gathered, covered, reached, equality
    
    def metric(self, _):
        """Returns results for one run.
        """

        model = self.sim()
        return model.instruments[0].metrics, model.instruments[1].metrics, model.instruments[2].metrics, model.instruments[3].metrics

    def sim(self):
        """Runs the model once. 
        """

        model = MajorCrimeIncidentResponse(
            world=self.world,
            search_radius_size = self.search_radius_size,
            number_of_responders = self.number_of_responders,
            pheromone_deposit = self.p,
            pheromone_decay = self.pheromone_decay,
            staggered_dispatch = self.staggered_dispatch,
            staggered_dispatch_responders = self.staggered_dispatch_responders,
            staggered_dispatch_delta = self.staggered_dispatch_delta,
            evolving_incident = self.evolving_incident,
            evolving_incident_growth_by = self.evolving_incident_growth_by,
            evolving_incident_growth_delta = self.evolving_incident_growth_delta,
            evolving_incident_regenerate_delta = self.evolving_incident_regenerate_delta,
            emerging_incident = self.emerging_incident,
            emerging_incident_k = self.emerging_incident_k,
            emerging_incident_n = self.emerging_incident_n,
            spreading_incident = self.spreading_incident,
            spreading_incident_ego_graph = self.spreading_incident_ego_graph,
            spreading_incident_threshold = self.spreading_incident_threshold,
            spreading_incident_probability = self.spreading_incident_probability,
            spreading_incident_time_multiplier = self.spreading_incident_time_multiplier
            )
        
        model.add_instrument(InformationGathering())
        model.add_instrument(SearchRadiusCoverage())
        model.add_instrument(ResponderReachedIncident())
        model.add_instrument(InformationEquality())
        model.run()
        return model
    
    def average_across_lengths(self, results):
        """Averages results per time step across simulations of varying length. 
        
        Aligns results per time step from simulations of varying lengths and averages them. 
        Shorter simulations are padded with NaN values which are ignored. 
        """

        max_time = max(len(sim) for sim in results)
        aligned_results = np.full((len(results), max_time), np.nan)

        for i, sim in enumerate(results):
            aligned_results[i, :len(sim)] = sim
            
        avg_results = np.nanmean(aligned_results, axis=0)
        sd_results = np.nanstd(aligned_results, axis=0)
            
        return avg_results, sd_results

    def convert_non_cum(self, gathered):
        """Converts cumulative information gathered into information hits.
        
        Converts results to show new information gathered per time step
        to measure continuous coverage of the incident. 
        """
        
        non_cumulatives = []

        for cum in gathered:
            non_cumulative_list = [cum[0]]

            for i in range(1, len(cum)):
                non_cum = cum[i] - cum[i - 1]
                non_cumulative_list.append(non_cum)

            non_cumulatives.append(non_cumulative_list)

        return non_cumulatives

class PheromoneAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        """Allows pheromone input as float, list or start:stop:step.
        """
        super().__init__(option_strings, dest, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        if ':' in values[0]:
            try:
                start, stop, step = map(float, values[0].split(':'))
                values = np.arange(start, stop + step, step)
                values = values[values <= stop]
            except ValueError:
                raise argparse.ArgumentTypeError("Pheromone range must be in the format start:stop:step")
        else:
            values = [float(x) for x in values]
        setattr(namespace, self.dest, values)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ensemble Simulations")
    parser.add_argument("-m", "--method", type=int, default=1, help="Method of simulation")
    parser.add_argument("-r", "--runs", type=int, default=100, help="Number of simulations")
    parser.add_argument("--number_of_responders", type=int, default=30, help="Number of responders")
    parser.add_argument("--pheromone_deposit", nargs="+", action=PheromoneAction, default=[0.3, 0.9], help="Pheromone deposit size (float, list or start:stop:step)")
    parser.add_argument("--pheromone_decay", type=float, default=0, help="Decay rate")
    
    parser.add_argument("--staggered_dispatch", action="store_true", help="Stagger the response")
    parser.add_argument("--staggered_dispatch_responders", type=int, default=10, help="Number of responders dispatched per interval")
    parser.add_argument("--staggered_dispatch_delta", type=int, default=20, help="Interval at which responders are dispatched")

    parser.add_argument("--search_radius_size", type=int, default=25, help="Amount of information points around the incident")
    parser.add_argument("--evolving_incident", action="store_true", help="Search radius grows and information regenerates")
    parser.add_argument("--evolving_incident_growth_by", type=int, default=0, help="Search radius grows by x euclidan distance")
    parser.add_argument("--evolving_incident_growth_delta", type=int, default=0, help="Interval at which the search radius grows")
    parser.add_argument("--evolving_incident_regenerate_delta", type=int, default=0, help="Time until information regenerates once gathered")
    
    parser.add_argument("--emerging_incident", action="store_true", help="Focus information gathering nearer to incident")
    parser.add_argument("--emerging_incident_k", type=float, default=0.2, help="Ensures reverse relationship and control pheromone adjustment")
    parser.add_argument("--emerging_incident_n", type=float, default=1.5, help="Controls the magnitude of effect by adding non-linearity")

    parser.add_argument("--spreading_incident", action="store_true", help="An additional incident occurs within the search radius")
    parser.add_argument("--spreading_incident_ego_graph", type=int, default=1, help="The amount of information gathered around each incident within x edges")
    parser.add_argument("--spreading_incident_threshold", type=float, default=0.5, help="The difference in information gathering that shifts focus to the other incident")
    parser.add_argument("--spreading_incident_probability", type=float, default=0.05, help="Base probability that an additional incident occurs")
    parser.add_argument("--spreading_incident_time_multiplier", type=float, default=0.001, help="Time multiplier increasing the chance that an additonal incident occurs")

    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed")
    parser.add_argument("-o", "--output_file", type=str, default="results.csv", help="Output file name to save results")
    return parser.parse_args()

def main():
    args = parse_arguments()
    output_file = args.output_file
    del args.output_file
    ensemble_args = vars(args)
    ensemble = Ensemble(**ensemble_args)
    df = ensemble.run()

    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f"results saved to {output_file}")

if __name__ == "__main__":
    main()