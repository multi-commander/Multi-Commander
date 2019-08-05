import json

from flask import jsonify


def get_cityflow_config(interval, seed, dir, roadnetFile, flowFile, rltrafficlight,savereplay):

    config = {'interval': interval, 'seed': seed, 'dir': dir, "roadnetFile": roadnetFile, "flowFile": flowFile,
              'rlTrafficLight': rltrafficlight, 'saveReplay': savereplay, "roadnetLogFile": "roadnet.json",
              "replayLogFile": "replay.txt"}
    jsonData = json.dumps(config)
    fileObject = open('./config/cityflow_config.json', 'w')
    fileObject.write(jsonData)
    fileObject.close()

#
# {"interval": 1.0,
#   "seed": 0,
#   "dir": "examples/",
#   "roadnetFile": "testcase_roadnet_1x1.json",
#   "flowFile": "testcase_flow_1x1.json",
#   "rlTrafficLight": true,
#   "saveReplay": false,
#   "roadnetLogFile": "roadnet.json",
#   "replayLogFile": "replay.txt"}


def main():
    print(get_cityflow_config(1,1,1,1,1,1,1))


if __name__ == "__main__":
    main()