import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--num_step", type=int, default=3600)
    return parser.parse_args()

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict ={}

    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    for intersection in roadnet["intersections"]:
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                     "end_lane": [],
                                                     "phase": [],
                                                     "phase_startLane_mapping": {},
                                                     "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        end_lane = []
        roadLink_lane_pair = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                end_lane.append(el)
                roadLink_lane_pair[ri].append((sl, el))

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))

        for phase_i in range(1, len(intersection["trafficLight"]["lightphases"])):
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            for ri in p["availableRoadLinks"]:
                lane_pair.extend(roadLink_lane_pair[ri])
                if roadLink_lane_pair[ri][0][0] not in start_lane:
                    start_lane.append(roadLink_lane_pair[ri][0][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = lane_pair

    return lane_phase_info_dict