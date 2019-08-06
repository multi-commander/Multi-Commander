import json
import os
import time
from multiprocessing import Process
import pickle
from config import DIC_AGENTS, DIC_ENVS
import sys


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log):
    path_to_pkl = os.path.join(path_to_log, "inter_0.pkl")
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)


def run_wrapper(dir, one_round, run_cnt, if_gui):
    model_dir = "model/" + dir
    records_dir = "records/" + dir
    model_round = one_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    with open(os.path.join(records_dir, "traffic_env.conf"), "r") as f:
        dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agent_name = dic_exp_conf["MODEL_NAME"]
    # TODO with intersection id
    agent = DIC_AGENTS[agent_name](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        cnt_round=0,  # useless
        # intersection_id=str(i)
    )
    if 1:
        agent.load_network(model_round)

        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                               path_to_work_directory=dic_path[
                                                                   "PATH_TO_WORK_DIRECTORY"],
                                                               dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0

        while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            for one_state in state:
                action = agent.choose_action(step_num, one_state)

                action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
        env.bulk_log()
        env.end_sumo()
        if not __debug__:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       model_round)
            # print("downsample", path_to_log)
            downsample(path_to_log)
            # print("end down")

    # except:
    #    pass
    # import sys
    # sys.stderr.write("fail to test model_%"%model_round)
    # raise SystemExit(1)

    return


def run_inference(dir, model_file_list, RUN_COUNT):
    # dir = 'cityflow'
    model_dir = "model/" + dir
    records_dir = "records/" + dir
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    with open(os.path.join(records_dir, "traffic_env.conf"), "r") as f:
        dic_traffic_env_conf = json.load(f)

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", 'log')
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    dic_traffic_env_conf["SAVEREPLAY"] = True

    env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                           path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                                                           dic_traffic_env_conf=dic_traffic_env_conf)

    state = env.reset()
    agents = [None] * dic_traffic_env_conf['NUM_AGENTS']

    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_exp_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            best_round=None,
            # intersection_id=str(i)
        )
        agent.load_network_(model_file_list[i])
        agents[i] = agent

    step_num = 0
    while step_num < RUN_COUNT:
        action_list = []
        for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
            for one_state in state:
                one_state['cur_phase'][0] = str(one_state['cur_phase'][0])
                action = agents[i].choose_action(step_num, one_state)
                action_list.append(action)

        next_state, reward, done, _ = env.step(action_list, False)

        state = next_state
        step_num += 1


def main(memo=None):
    # run name
    if not memo:
        memo = "New_sumo/pipeline_template/"

    # test run_count
    run_cnt = 3600

    # add the specific rounds in the given_round_list, like [150, 160]
    # if none, test all the round
    given_round_list = [63]

    given_traffic_list = [
        # "cross.2phases_rou01_equal_650.xml",
        # "cross.2phases_rou01_equal_600.xml",
        # "cross.2phases_rou01_equal_550.xml",
        # "cross.2phases_rou01_equal_500.xml",
        # "cross.2phases_rou01_equal_450.xml",
        # "cross.2phases_rou01_equal_400.xml",
        # "cross.2phases_rou01_equal_350.xml",
        # "cross.2phases_rou01_equal_300.xml",
    ]

    if_gui = True

    multi_process = True
    n_workers = 100
    process_list = []
    for traffic in os.listdir("records/" + memo):
        print(traffic)
        if not ".xml" in traffic:
            continue

        if traffic != "cross.2phases_rou01_equal_100.xml_12_12_19_49_03":
            continue
        test_round_dir = os.path.join("records", memo, traffic, "test_round")
        if os.path.exists(test_round_dir):
            print("exist")
            # continue
        # if traffic[0:-15] not in given_traffic_list:
        #    continue

        work_dir = os.path.join(memo, traffic)

        if given_round_list:
            for one_round in given_round_list:
                _round = "round_" + str(one_round)
                if multi_process:
                    p = Process(target=run_wrapper, args=(work_dir, _round, run_cnt, if_gui))
                    process_list.append(p)
                else:
                    run_wrapper(work_dir, _round, run_cnt, if_gui)
        else:
            train_round_dir = os.path.join("records", memo, traffic, "train_round")
            for one_round in os.listdir(train_round_dir):
                if "round" not in one_round:
                    continue

                if multi_process:
                    p = Process(target=run_wrapper, args=(work_dir, one_round, run_cnt, if_gui))
                    process_list.append(p)
                else:
                    run_wrapper(work_dir, one_round, run_cnt, if_gui)

    if multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < n_workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < n_workers:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for p in list_cur_p:
            p.join()


if __name__ == "__main__":
    # main()
    model_file_list = ['./model/test/round_216_inter_0.h5', './model/test/round_216_inter_1.h5',
                       './model/test/round_216_inter_2.h5', './model/test/round_216_inter_3.h5',
                       './model/test/round_216_inter_4.h5', './model/test/round_216_inter_5.h5']

    dir = 'test/anon_1_6_300_0.3_synthetic.json_08_05_19_55_50'
    NUM_COUNT = 700

    run_inference(dir, model_file_list, NUM_COUNT)
