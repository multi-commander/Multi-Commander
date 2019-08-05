import config
import copy
from pipeline import Pipeline
import os
import time
from multiprocessing import Process
import argparse
import os
import matplotlib
# matplotlib.use('TkAgg')

multi_process = True

PRETRAIN=False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='test')
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='1_6')
    parser.add_argument("--volume", type=int, default=300)
    parser.add_argument("--suffix", type=str, default="0.3")
    parser.add_argument("--mod", type=str, default="SimpleDQN")
    parser.add_argument("--lane", type=int, default=1)
    parser.add_argument("-syn", action="store_true",default=True)
    parser.add_argument("-uniform", action="store_true",default=False)
    parser.add_argument("-jinan", action="store_true",default=False)
    parser.add_argument("-beaver", action="store_true",default=False)

    parser.add_argument("--mode", type=int,default=3)

    parser.add_argument("--cnt",type=int, default=3600)
    parser.add_argument("--gen",type=int, default=1)
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers",type=int, default=8)
    parser.add_argument("-onemodel",action="store_true", default=False)
    parser.add_argument("--visible_gpu", type=str, default="")
    PRETRAIN = False

    return parser.parse_args()


def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path
                   )
    global multi_process
    ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return



def main(args = None):

    ENVIRONMENT = ["sumo", "anon"][1]
    num_rounds = 300

    separate_test = False
    sparse_test = False
    transfer = False
    if args.uniform:
        traffic_file_list = []
        args.lane = 1
        args.cnt = 3600
        args.road_net = "1_6"
        data_path = os.path.join("data", "template_s", "1_6")
        traffic_file_list = [i for i in os.listdir(os.path.join(os.getcwd(),data_path)) if 'roadnet' not in i]
        num_rounds = 300
    elif args.jinan:
        args.lane = 3
        args.cnt = 3600
        args.road_net = "1_3"
        traffic_file_list = ["anon_6_7_real.json",
                             "anon_16_17_real.json"]
        num_rounds = 500
    elif args.beaver:
        args.lane = 3
        args.cnt = 3600
        args.road_net = "1_5"
        traffic_file_list = ["anon_1_5_700_0.6_beaver.json"]
        num_rounds = 500
        # transfer = True
    elif args.all:
        traffic_file_list = []
        # traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s.json" %(v,r) for v in [300,700] for r in [0.3,0.6]]
        traffic_file_list.extend([ENVIRONMENT+"_"+args.road_net+"_%d_%s_uni.json" %(v,r) for v in [300,700] for r in [0.3,0.6]])
        num_rounds = 500
    elif args.syn:
        # separate_test = True
        # sparse_test = True
        # traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s_%d" % (v, r, j) for v in [500, 700] for r in
        #                      [0.3, 0.6] for j in [1, 3]]
        # cnt = 10800
        traffic_file_list = [ENVIRONMENT + "_" + args.road_net + "_%d_%s_synthetic.json" % (v, r) for v in [300,700] for r in
                             [0.3,0.6]]
        # road_net = "3_3"
        args.lane = 3
        num_rounds = 600
        # transfer = True
    elif args.road_net == '3_3':
        traffic_file_list = [ENVIRONMENT + "_" + args.road_net + "_%d_%s.json" % (v, r) for v in [300,500,700] for r in
                             [0.6]]
        args.lane = 1
        num_rounds = 700
    elif args.road_net == "16_1":
        traffic_file_list = ["anon_16_1_300_newyork_real_3.json",
                             "anon_16_1_300_newyork_real_2.json",
                             "anon_16_1_300_newyork_real_1.json",
                             "anon_16_1_300_newyork_real_4.json"]
        args.lane = 3
        num_rounds = 700
    else:
        traffic_file_list = ["{0}_{1}_{2}_{3}.json".format(ENVIRONMENT, args.road_net, args.volume, args.suffix)]

    NUM_COL = int(args.road_net.split('_')[0])
    NUM_ROW = int(args.road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    process_list = []
    n_workers = args.workers     #len(traffic_file_list)
    multi_process = True

    for traffic_file in traffic_file_list:  # [ind_arg:ind_arg+1]:


        
        dic_exp_conf_extra = {

            "RUN_COUNTS": args.cnt,
            "MODEL_NAME": args.mod,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),

            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": args.gen,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,
#

            "PRETRAIN": False,#
            "PRETRAIN_MODEL_NAME":args.mod,
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": False,
            "SPARSE_TEST": sparse_test,
        }

        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 1000,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,
            "PRIORITY": False,

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,
        }

        dic_traffic_env_conf_extra = {

            "ONE_MODEL": args.onemodel,
            "TRANSFER": transfer,

            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "IF_GUI": args.gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": 4,

            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,
            "SEPARATE_TEST": separate_test,

            "NEIGHBOR": False,
            "MODEL_NAME": args.mod,

            "SAVEREPLAY": False,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": args.volume,

            "ROADNET_FILE": "roadnet_{0}.json".format(args.road_net),

            "TRAFFIC_SEPARATE":traffic_file,

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle"
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),
                    D_LANE_NUM_VEHICLE_DOWNSTREAM=(4,),
                    D_DELTA_LANE_NUM_VEHICLE=(4,),
                    D_NUM_TOTAL_VEH = (1,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(4,),

                    D_ADJACENCY_MATRIX=(2,),

                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0] # 'NLSL',
                },
                "anon": {
                    # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    4: [0, 0, 0, 0, 1, 0, 1, 0] # 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                },
            },

            "list_lane_order": ["WL", "WT", "EL", "ET", "SL", "ST", "NL", "NT"],

            "PHASE_LIST": [
                'WT_ET',
                'NT_ST',
                'WL_EL',
                'NL_SL',
            ],

        }

        ## ==================== multi_phase ====================
        if args.lane ==2:
            template = "template_ls"
        elif args.lane ==1:
            template = "template_s"
            dic_traffic_env_conf_extra["PHASE"] = {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                },
                "anon": {
                    1: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                },
                }
            dic_traffic_env_conf_extra["LANE_NUM"] ={
                "LEFT": 0,
                "RIGHT": 0,
                "STRAIGHT": 1
            }
        elif args.lane == 3:
            if args.jinan:
                template = "template_sss"
                dic_traffic_env_conf_extra["LANE_NUM"] = {
                    "LEFT": 0,
                    "RIGHT": 0,
                    "STRAIGHT": 3
                }

            else:
                template = "template_lsr"
        else:
            raise ValueError

        # ================== mode ============== 0-IRL 1-segmentRL 2-allFeatureRL 3-NewRL

        if args.mode == 3:
            # arterial presslight
            dic_traffic_env_conf_extra["LIST_STATE_FEATURE"] = ["cur_phase", "coming_vehicle", "leaving_vehicle"]
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"] = {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": 0,#-0.25,
                "pressure": -0.25
            }
            dic_traffic_env_conf_extra["FAST_COMPUTE"] = False
            # dic_agent_conf_extra["PRIORITY"] = True

        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)


        print(traffic_file)
        prefix_intersections = str(args.road_net)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", args.memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_TRANSFER_MODEL":os.path.join("data/template_lsr",args.road_net,str(args.mode)),

            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", args.memo)
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(args.mod.upper())),
                                      dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

        # TODO add agent_conf for different agents
        # deploy_dic_agent_conf_all = [deploy_dic_agent_conf for i in range(deploy_dic_traffic_env_conf["NUM_AGENTS"])]

        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)


    return args.memo


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main(args)

