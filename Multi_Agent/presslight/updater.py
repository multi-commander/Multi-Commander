import pickle
import os
from config import DIC_AGENTS
import pandas as pd
import shutil
import pandas as pd
import time
from multiprocessing import Pool
import traceback
import random
import numpy as np


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        print("Number of agents: ", dic_traffic_env_conf['NUM_AGENTS'])

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]

            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)



    def load_sample_with_forget(self, i):

        sample_set = []
        try:
            if self.dic_exp_conf["PRETRAIN"]:
                    sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                "train_round", "total_samples" + ".pkl"), "rb")
            elif self.dic_exp_conf["AGGREGATE"]:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                                "aggregate_samples.pkl"), "rb")
            else:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
                    ind_end = len(sample_set)
                    ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                    memory_after_forget = sample_set[ind_sta: ind_end]
                    sample_set = memory_after_forget

            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i %100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set


    def load_sample_for_agents(self):
        start_time = time.time()
        print("Start load samples at", start_time)
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            sample_set = self.load_sample_with_forget(i)
            self.agents[i].prepare_Xs_Y(sample_set, self.dic_exp_conf)

        print("------------------Load samples time: ", time.time()-start_time)

    def sample_set_to_sample_gcn_df(self,sample_set):
        print("make results")
        samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
        samples_set_df = samples_set_df.set_index(['time','generator'])
        samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
        samples_set_df.drop(['state','action','next_state','inst_reward','reward'], axis=1, inplace=True)
        self.sample_set_list.append(samples_set_df)


    def update_network(self,i):
        print('update agent %d'%i)
        self.agents[i].train_network(self.dic_exp_conf)
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0]))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0])),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate.h5")
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                self.agents[i].save_network("round_{0}".format(self.cnt_round))

        else:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                       self.agents[i].intersection_id))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                   self.agents[i].intersection_id)),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate_inter_{0}.h5".format(self.agents[i].intersection_id))
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(self.agents[i].intersection_id)))
            else:
                self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round,self.agents[i].intersection_id))

    def update_network_for_agents(self):
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.update_network(0)
        else:
            print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
            for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                self.update_network(i)



