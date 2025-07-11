import traci
import sys
from constraint import Problem, BacktrackingSolver
import numpy as np
from sumoenv import SumoEnv 

class TrafficLightCSP:
    def __init__(self):
        self.env = SumoEnv(label='csp_sim', gui_f=True) # Mengatur gui_f=True untuk visualisasi
        self.tl_id = "gneJ00" 
        self.ns_lanes = ['-gneE0_0', '-gneE0_1', '-gneE0_2', '-gneE2_0', '-gneE2_1', '-gneE2_2']
        # Perbaikan typo: mengubah '-gneE1_1' kedua menjadi '-gneE1_2'
        self.ew_lanes = ['-gneE1_0', '-gneE1_1', '-gneE1_2', '-gneE3_0', '-gneE3_1', '-gneE3_2']
        
        self.min_green = 20
        self.max_green = 60
        self.yellow_time = 5
        self.red_time = 0 

        self.step = 0
        self.total_vehicles_departed = 0
        self.total_waiting_time = 0.0 # This will store sum of total waiting times for *arrived* vehicles
        self.vehicle_travel_times = {}
        self.vehicle_departure_times = {}
        # New: Dictionary to accumulate waiting time for each *active* vehicle
        self.accumulated_waiting_time_per_veh = {} 
        
        # Mengubah variabel untuk menghitung total mobil di jalur
        self.current_ns_waiting_time = 0.0 # instantaneous waiting time sum for lanes
        self.current_ew_waiting_time = 0.0 # instantaneous waiting time sum for lanes
        self.current_ns_vehicle_count = 0 
        self.current_ew_vehicle_count = 0 
        
        # Bobot untuk fungsi biaya (tuning ini sangat penting!)
        self.ns_waiting_time_weight = 1.0 
        self.ew_waiting_time_weight = 1.0 
        self.ns_vehicle_count_weight = 0.5 
        self.ew_vehicle_count_weight = 0.5 
        self.imbalance_penalty = 10.0 
        self.over_max_green_penalty = 5.0 
        # NEW: Penalty for longer green light durations
        self.green_duration_penalty_weight = 0.01 # Adjust this value as needed


    def _get_current_lane_metrics(self):
        # Mendapatkan waktu tunggu saat ini untuk setiap arah
        # This is instantaneous waiting time for halting vehicles on specified lanes
        self.current_ns_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in self.ns_lanes)
        self.current_ew_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in self.ew_lanes)
        
        # Mendapatkan jumlah total mobil di jalur untuk setiap arah
        self.current_ns_vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in self.ns_lanes)
        self.current_ew_vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in self.ew_lanes)


    def calculate_cost(self, ns_vehicle_count, ew_vehicle_count, ns_green, ew_green):
        cost = 0
        # Tujuan: Meminimalkan waktu tunggu dan jumlah mobil di jalur
        # Menggunakan waktu tunggu saat ini yang sudah didapatkan (instantaneous lane waiting time)
        cost += self.current_ns_waiting_time * self.ns_waiting_time_weight
        cost += self.current_ew_waiting_time * self.ew_waiting_time_weight
        
        # Menambahkan bobot untuk jumlah mobil di jalur
        cost += ns_vehicle_count * self.ns_vehicle_count_weight
        cost += ew_vehicle_count * self.ew_vehicle_count_weight

        # Kendala/Penalti untuk ketidakseimbangan yang tidak diatasi dengan benar
        if ns_vehicle_count > ew_vehicle_count and ns_green < ew_green:
            cost += (ew_green - ns_green) * self.imbalance_penalty
        elif ew_vehicle_count > ns_vehicle_count and ew_green < ns_green:
            cost += (ns_green - ew_green) * self.imbalance_penalty
        
        # NEW: Add a small penalty for longer green light durations
        cost += ns_green * self.green_duration_penalty_weight
        cost += ew_green * self.green_duration_penalty_weight
            
        return cost


    def run_simulation(self, total_steps):
        try:
            self.env.reset()
            traci.trafficlight.setPhase(self.tl_id, 0) 
            traci.trafficlight.setPhaseDuration(self.tl_id, self.min_green)

            while self.step < total_steps:
                traci.simulationStep()
                self.step += 1

                # Track departed vehicles for departure times
                for veh_id in traci.simulation.getDepartedIDList():
                    self.vehicle_departure_times[veh_id] = traci.simulation.getTime()

                # Accumulate waiting time for *active* vehicles at each step
                # Iterate over all vehicles currently in the simulation
                for veh_id in traci.vehicle.getIDList(): 
                    self.accumulated_waiting_time_per_veh[veh_id] = \
                        self.accumulated_waiting_time_per_veh.get(veh_id, 0) + traci.vehicle.getWaitingTime(veh_id)

                # For vehicles that arrived at their destination
                for veh_id in traci.simulation.getArrivedIDList():
                    if veh_id in self.vehicle_departure_times:
                        self.total_vehicles_departed += 1
                        travel_time = traci.simulation.getTime() - self.vehicle_departure_times[veh_id]
                        self.vehicle_travel_times[veh_id] = travel_time
                        
                        # Add the total accumulated waiting time for this vehicle
                        if veh_id in self.accumulated_waiting_time_per_veh:
                            self.total_waiting_time += self.accumulated_waiting_time_per_veh[veh_id]
                            del self.accumulated_waiting_time_per_veh[veh_id] # Clean up to save memory
                        # else: This vehicle might have departed and arrived within the same step or an edge case

                # Mendapatkan metrik jumlah mobil dan waktu tunggu saat ini (instantaneous)
                self._get_current_lane_metrics()

                if self.step % 50 == 0 or self.step == 1:
                    print(f"------------------------------")
                    print(f"Step {self.step}:")
                    print(f"  NS Vehicle Count: {self.current_ns_vehicle_count}, NS Waiting Time: {self.current_ns_waiting_time:.2f}")
                    print(f"  EW Vehicle Count: {self.current_ew_vehicle_count}, EW Waiting Time: {self.current_ew_waiting_time:.2f}")

                    problem = Problem(BacktrackingSolver())

                    problem.addVariable('ns_green', range(self.min_green, self.max_green + 1))
                    problem.addVariable('ew_green', range(self.min_green, self.max_green + 1))

                    problem.addConstraint(lambda ns: ns >= self.min_green, ['ns_green'])
                    problem.addConstraint(lambda ew: ew >= self.min_green, ['ew_green'])

                    problem.addConstraint(lambda ns: ns <= self.max_green, ['ns_green'])
                    problem.addConstraint(lambda ew: ew <= self.max_green, ['ew_green'])

                    if self.current_ns_vehicle_count > self.current_ew_vehicle_count:
                        problem.addConstraint(lambda ns, ew: ns >= ew, ['ns_green', 'ew_green'])
                        problem.addConstraint(lambda ns, ew: ns - ew >= 5, ['ns_green', 'ew_green'])
                    elif self.current_ew_vehicle_count > self.current_ns_vehicle_count:
                        problem.addConstraint(lambda ns, ew: ew >= ns, ['ns_green', 'ew_green'])
                        problem.addConstraint(lambda ew, ns: ew - ns >= 5, ['ew_green', 'ns_green'])
                    else:
                        problem.addConstraint(lambda ns, ew: abs(ns - ew) <= 10, ['ns_green', 'ew_green'])

                    problem.addConstraint(lambda ns, ew: ns >= self.min_green and ew >= self.min_green, ['ns_green', 'ew_green'])

                    # Add the objective function constraint
                    problem.addConstraint(
                        lambda ns_green, ew_green: self.calculate_cost(
                            self.current_ns_vehicle_count, self.current_ew_vehicle_count, 
                            ns_green, ew_green
                        ) == min(
                            self.calculate_cost(self.current_ns_vehicle_count, self.current_ew_vehicle_count, ng, eg)
                            for ng in range(self.min_green, self.max_green + 1)
                            for eg in range(self.min_green, self.max_green + 1)
                            # Filter solutions that meet other constraints for accurate min_cost calculation
                            if ( (self.current_ns_vehicle_count > self.current_ew_vehicle_count and ng >= eg and ng - eg >= 5) or
                                 (self.current_ew_vehicle_count > self.current_ns_vehicle_count and eg >= ng and eg - ng >= 5) or
                                 (self.current_ns_vehicle_count == self.current_ew_vehicle_count and abs(ng - eg) <= 10) )
                            and ng >= self.min_green and eg >= self.min_green and ng <= self.max_green and eg <= self.max_green
                        ),
                        ['ns_green', 'ew_green']
                    )

                    solutions = problem.getSolutions()

                    optimal_solution = None
                    min_cost = float('inf')

                    if solutions:
                        for sol in solutions:
                            cost = self.calculate_cost(self.current_ns_vehicle_count, self.current_ew_vehicle_count, sol['ns_green'], sol['ew_green'])
                            if cost < min_cost:
                                min_cost = cost
                                optimal_solution = sol
                        
                        green_ns_final = optimal_solution['ns_green']
                        green_ew_final = optimal_solution['ew_green']
                        print(f"  Green NS Final: {green_ns_final}s, Green EW Final: {green_ew_final}s")
                        print(f"  Optimal CSP solution found with cost: {min_cost:.2f}")

                        traci.trafficlight.setPhase(self.tl_id, 0) 
                        traci.trafficlight.setPhaseDuration(self.tl_id, green_ns_final)
                        for _ in range(green_ns_final):
                            traci.simulationStep()
                            self.step += 1
                        
                        traci.trafficlight.setPhase(self.tl_id, 1)
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_time)
                        for _ in range(self.yellow_time):
                            traci.simulationStep()
                            self.step += 1

                        traci.trafficlight.setPhase(self.tl_id, 2)
                        traci.trafficlight.setPhaseDuration(self.tl_id, green_ew_final)
                        for _ in range(green_ew_final):
                            traci.simulationStep()
                            self.step += 1

                        traci.trafficlight.setPhase(self.tl_id, 3)
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_time)
                        for _ in range(self.yellow_time):
                            traci.simulationStep()
                            self.step += 1

                    else:
                        print(f"  No optimal CSP solution found for step {self.step}. Using default green times.")
                        green_ns_final = self.min_green
                        green_ew_final = self.min_green
                        print(f"  Green NS Final: {green_ns_final}s, Green EW Final: {green_ew_final}s")
                        
                        traci.trafficlight.setPhase(self.tl_id, 0)
                        traci.trafficlight.setPhaseDuration(self.tl_id, green_ns_final)
                        for _ in range(green_ns_final):
                            traci.simulationStep()
                            self.step += 1
                        
                        traci.trafficlight.setPhase(self.tl_id, 1)
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_time)
                        for _ in range(self.yellow_time):
                            traci.simulationStep()
                            self.step += 1

                        traci.trafficlight.setPhase(self.tl_id, 2)
                        traci.trafficlight.setPhaseDuration(self.tl_id, green_ew_final)
                        for _ in range(green_ew_final):
                            traci.simulationStep()
                            self.step += 1

                        traci.trafficlight.setPhase(self.tl_id, 3)
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_time)
                        for _ in range(self.yellow_time):
                            traci.simulationStep()
                            self.step += 1


            if self.total_vehicles_departed > 0:
                avg_waiting_time = self.total_waiting_time / self.total_vehicles_departed
                total_travel_time = sum(self.vehicle_travel_times.values())
                avg_travel_time = total_travel_time / self.total_vehicles_departed if self.total_vehicles_departed > 0 else 0
                throughput = self.total_vehicles_departed / self.step if self.step > 0 else 0
                
                print(f"\n--- Simulation Summary (CSP Adaptive Traffic Light Optimized by Vehicle Count) ---")
                print(f"Simulation ended at step {self.step}. Total vehicles departed: {self.total_vehicles_departed}")
                print(f"Total waiting time: {self.total_waiting_time:.2f}s, Average waiting time per vehicle: {avg_waiting_time:.2f}s")
                print(f"Total travel time: {total_travel_time:.2f}s, Average travel time per vehicle: {avg_travel_time:.2f}s")
                print(f"Throughput: {throughput:.4f} vehicles/step")
            else:
                print("No vehicles departed during the simulation.")
        except Exception as e:
            print(f"Simulation terminated with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.env.close()
            print("TraCI connection closed successfully")


if __name__ == "__main__":
    traffic_light = TrafficLightCSP()
    traffic_light.run_simulation(total_steps=500)