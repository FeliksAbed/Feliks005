import traci
import sys
import numpy as np
from sumoenv import SumoEnv # Pastikan sumoenv.py ada dan benar

class TrafficLightStatic:
    def __init__(self):
        self.env = SumoEnv(label='static_sim', gui_f=True) # Label yang berbeda untuk sim statis
        self.tl_id = "gneJ00"
        self.ns_lanes = ['-gneE0_0', '-gneE0_1', '-gneE0_2', '-gneE2_0', '-gneE2_1', '-gneE2_2']
        self.ew_lanes = ['-gneE1_0', '-gneE1_1', '-gneE1_2', '-gneE3_0', '-gneE3_1', '-gneE3_2']
        
        # Pengaturan lampu lalu lintas statis
        self.green_ns = 60 # Durasi lampu hijau untuk arah North-South
        self.green_ew = 60 # Durasi lampu hijau untuk arah East-West
        self.yellow_time = 5
        self.red_time = 0 # Asumsi TraCI mengelola waktu merah antar fase dengan baik, atau bisa 2-3s jika diperlukan

        self.step = 0
        self.total_vehicles_departed = 0
        self.total_waiting_time = 0.0
        self.vehicle_travel_times = {}
        self.vehicle_departure_times = {}
        
        # Metrik untuk logging
        self.current_ns_waiting_time = 0.0
        self.current_ew_waiting_time = 0.0
        self.current_ns_queue_length = 0
        self.current_ew_queue_length = 0

        # Batas langkah simulasi untuk perbandingan yang adil
        self.max_simulation_steps = 500 # === PASTIKAN NILAI INI SAMA DENGAN CSP ===

    def _update_vehicle_metrics(self):
        """Updates vehicle departure and travel times."""
        # Update departure times for newly arrived vehicles in the simulation
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in self.vehicle_departure_times:
                self.vehicle_departure_times[veh_id] = self.step

        # Calculate travel times for vehicles that have arrived at their destination
        arrived_vehicles = traci.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            if veh_id in self.vehicle_departure_times:
                travel_time = self.step - self.vehicle_departure_times[veh_id]
                self.vehicle_travel_times[veh_id] = travel_time
                del self.vehicle_departure_times[veh_id]

    def _get_current_lane_metrics(self):
        """
        Collects real-time queue lengths (halting vehicles) and average waiting times per lane.
        This runs at the beginning of each cycle to get current state (for logging purposes only in static).
        """
        ns_queue = 0
        ew_queue = 0
        ns_waiting_sum = 0.0
        ew_waiting_sum = 0.0
        ns_vehicle_count_for_avg_wait = 0 
        ew_vehicle_count_for_avg_wait = 0

        for lane in self.ns_lanes:
            ns_queue += traci.lane.getLastStepHaltingNumber(lane)
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ns_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ns_vehicle_count_for_avg_wait += 1
        
        for lane in self.ew_lanes:
            ew_queue += traci.lane.getLastStepHaltingNumber(lane)
            current_lane_vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in current_lane_vehicles:
                ew_waiting_sum += traci.vehicle.getWaitingTime(veh_id)
                ew_vehicle_count_for_avg_wait += 1

        self.current_ns_waiting_time = (ns_waiting_sum / ns_vehicle_count_for_avg_wait) if ns_vehicle_count_for_avg_wait > 0 else 0.0
        self.current_ew_waiting_time = (ew_waiting_sum / ew_vehicle_count_for_avg_wait) if ew_vehicle_count_for_avg_wait > 0 else 0.0
        
        self.current_ns_queue_length = ns_queue
        self.current_ew_queue_length = ew_queue


    def _run_phase(self, phase_duration, phase_id):
        """Helper to run a traffic light phase and update metrics and log."""
        self.env.set_traffic_light_phase(phase_id, phase_duration)
        for _ in range(int(phase_duration)):
            # === PENTING: Pengecekan untuk menghentikan simulasi jika mencapai batas langkah ===
            if self.step >= self.max_simulation_steps:
                break 
            
            self.env.simulation_step()
            self._update_vehicle_metrics()
            
            total_halting_vehicles_current_step = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.ns_lanes + self.ew_lanes)
            current_total_waiting_time_step = self.env.get_waiting_time() # Total waiting time at intersection for this step

            self.total_waiting_time += current_total_waiting_time_step
            
            # Mendapatkan metrik untuk logging
            self._get_current_lane_metrics()

            with open('static_queue_length.txt', 'a') as f:
                f.write(f"{self.step},{total_halting_vehicles_current_step},{current_total_waiting_time_step},"
                        f"{self.current_ns_waiting_time:.2f},{self.current_ew_waiting_time:.2f}\n")
            self.step += 1

    def run(self):
        self.env.reset()
        # Header untuk file log statis
        with open('static_queue_length.txt', 'w') as f:
            f.write("step,queue_length,waiting_time,ns_avg_waiting_time,ew_avg_waiting_time\n")

        try:
            # === PENTING: Mengubah kondisi while loop untuk berhenti pada jumlah langkah yang sama ===
            while self.step < self.max_simulation_steps:
                self.total_vehicles_departed = len(self.vehicle_travel_times)
                print(f"Total vehicles departed: {self.total_vehicles_departed}")
                print(f"Step {self.step}: Static timing - NS: {self.green_ns}s, EW: {self.green_ew}s")
                
                # Jalankan fase lalu lintas statis
                self._run_phase(self.green_ns, 0) # Phase 0: NS Green
                if self.step >= self.max_simulation_steps: break # Pengecekan setelah fase
                
                self._run_phase(self.yellow_time, 1) # Phase 1: Yellow NS
                if self.step >= self.max_simulation_steps: break
                
                self._run_phase(self.green_ew, 2) # Phase 2: EW Green
                if self.step >= self.max_simulation_steps: break
                
                self._run_phase(self.yellow_time, 3) # Phase 3: Yellow EW
                if self.step >= self.max_simulation_steps: break

            # Hitung dan cetak metrik performa akhir
            if self.total_vehicles_departed > 0:
                avg_waiting_time = self.total_waiting_time / self.total_vehicles_departed
                total_travel_time = sum(self.vehicle_travel_times.values())
                avg_travel_time = total_travel_time / self.total_vehicles_departed if self.total_vehicles_departed > 0 else 0
                throughput = self.total_vehicles_departed / self.step if self.step > 0 else 0
                # Efisiensi siklus tidak relevan di sini karena tidak ada "siklus" adaptif
                
                print(f"\n--- Simulation Summary (Static Traffic Light) ---")
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
            sys.stdout.flush()

if __name__ == "__main__":
    static_sim = TrafficLightStatic()
    static_sim.run()