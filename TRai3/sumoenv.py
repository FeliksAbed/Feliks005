import os
import sys
import numpy as np
import traci

# Setup SUMO tools path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class SumoEnv:
    place_len = 7.5
    place_offset = 8.50
    lane_len = 10
    lane_ids = [
        '-gneE0_0','-gneE0_1','-gneE0_2',
        '-gneE1_0','-gneE1_1','-gneE1_2',
        '-gneE2_0','-gneE2_1','-gneE2_2',
        '-gneE3_0','-gneE3_1','-gneE3_2'
    ]

    def __init__(self, label='default', gui_f=False):
        self.label = label
        self.ncars = 0
        exe = 'sumo-gui' if gui_f else 'sumo'
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', exe)
        self.sumoCmd = [sumoBinary, '-c', 'intersection.sumocfg']
    
    def reset(self):
        self.ncars = 0

        # Cegah error jika traci sudah terhubung sebelumnya
        if traci.isLoaded():
            try:
                traci.close()
            except:
                pass

        traci.start(self.sumoCmd, label=self.label)
        traci.trafficlight.setProgram('gneJ00', '0')  # pastikan program id = '0'
        traci.simulationStep()
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.lane_len * 12 + 4, dtype=np.float32)
        for ilane in range(12):
            lane_id = self.lane_ids[ilane]
            cars = traci.lane.getLastStepVehicleIDs(lane_id)
            for icar in cars:
                xcar, ycar = traci.vehicle.getPosition(icar)
                if ilane < 3:
                    pos = (ycar - self.place_offset) / self.place_len
                elif ilane < 6:
                    pos = (xcar - self.place_offset) / self.place_len
                elif ilane < 9:
                    pos = (-ycar - self.place_offset) / self.place_len
                else:
                    pos = (-xcar - self.place_offset) / self.place_len
                if pos > self.lane_len - 1.:
                    continue
                pos = np.clip(pos, 0., self.lane_len - 1. - 1e-6)
                ipos = int(pos)
                state[ilane * self.lane_len + ipos] += 1. - pos + ipos
                state[ilane * self.lane_len + ipos + 1] += pos - ipos

        phase = traci.trafficlight.getPhase('gneJ00')
        state[self.lane_len * 12 : self.lane_len * 12 + 4] = np.eye(4)[phase]
        return state

    def get_waiting_time(self):
        return sum(traci.lane.getWaitingTime(lane_id) for lane_id in self.lane_ids)

    def set_traffic_light_phase(self, phase, duration):
        traci.trafficlight.setPhase('gneJ00', phase)
        traci.trafficlight.setPhaseDuration('gneJ00', duration)

    def simulation_step(self):
        traci.simulationStep()
        self.ncars += traci.simulation.getDepartedNumber()

    def close(self):
        if traci.isLoaded():
            traci.close()
