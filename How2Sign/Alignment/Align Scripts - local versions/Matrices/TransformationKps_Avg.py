import json
import numpy as np

R_rarms_def = []
t_rarms_def = []
R_larms_def = []
t_larms_def = []
R_body_def = []
t_body_def = []
R_head_def = []
t_head_def = []
R_codo_def = []
t_codo_def = []

start_manual = 46
for iters in range(10):
    adding= 5 + iters*5
    index = start_manual + adding

    #load frame matrices
    with open('../FrameMatrices/R_rarm'+ str(index) +'.json', 'r', encoding='utf8') as f:
        R_rarms = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/t_rarm'+ str(index) +'.json', 'r', encoding='utf8') as f:
        t_rarms = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/R_larm'+ str(index) +'.json', 'r', encoding='utf8') as f:
        R_larms = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/t_larm'+ str(index) +'.json', 'r', encoding='utf8') as f:
        t_larms = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/R_body'+ str(index) +'.json', 'r', encoding='utf8') as f:
        R_body = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')
    with open('../FrameMatrices/t_body'+ str(index) +'.json', 'r', encoding='utf8') as f:
        t_body = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/R_codo'+ str(index) +'.json', 'r', encoding='utf8') as f:
        R_codo = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')
    with open('../FrameMatrices/t_codo'+ str(index) +'.json', 'r', encoding='utf8') as f:
        t_codo = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    with open('../FrameMatrices/R_head'+ str(index) +'.json', 'r', encoding='utf8') as f:
        R_head = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')
    with open('../FrameMatrices/t_head'+ str(index) +'.json', 'r', encoding='utf8') as f:
        t_head = json.loads('[' + f.read().replace('}\n{', '},\n{') + ']')

    #adding all the matrices
    if iters == 0:
        R_rarms_def =  np.array(R_rarms)
        t_rarms_def = np.array(t_rarms)
        R_larms_def = np.array(R_larms)
        t_larms_def =  np.array(t_larms)
        R_body_def =  np.array(R_body)
        t_body_def = np.array(t_body)
        R_head_def = np.array(R_head)
        t_head_def = np.array(t_head)
        R_codo_def = np.array(R_codo)
        t_codo_def = np.array(t_codo)
    
    else:
        R_rarms_def = np.array(R_rarms_def) + np.array(R_rarms)
        t_rarms_def = np.array(t_rarms_def) + np.array(t_rarms)
        R_larms_def = np.array(R_larms_def) + np.array(R_larms)
        t_larms_def = np.array(t_larms_def) + np.array(t_larms)
        R_body_def = np.array(R_body_def) + np.array(R_body)
        t_body_def = np.array(t_body_def) + np.array(t_body)
        R_head_def = np.array(R_head_def) + np.array(R_head)
        t_head_def = np.array(t_head_def) + np.array(t_head)
        R_codo_def = np.array(R_codo_def) + np.array(R_codo)
        t_codo_def = np.array(t_codo_def) + np.array(t_codo)


#obtaining the average matrix
R_rarms_def = np.array(R_rarms_def)/10
t_rarms_def = np.array(t_rarms_def)/10
R_larms_def = np.array(R_larms_def)/10
t_larms_def = np.array(t_larms_def)/10
R_body_def = np.array(R_body_def)/10
t_body_def = np.array(t_body_def)/10
R_head_def = np.array(R_head_def)/10
t_head_def = np.array(t_head_def)/10
R_codo_def = np.array(R_codo_def)/10
t_codo_def = np.array(t_codo_def)/10

#saving final transformations
json_object = json.dumps(R_rarms_def.tolist()) 
with open('../FinalMatrices/R_rarms.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(t_rarms_def.tolist()) 
with open('../FinalMatrices/t_rarms.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(R_larms_def.tolist()) 
with open('../FinalMatrices/R_larms.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(t_larms_def.tolist()) 
with open('../FinalMatrices/t_larms.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(R_head_def.tolist()) 
with open('../FinalMatrices/R_head.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(t_head_def.tolist()) 
with open('../FinalMatrices/t_head.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(R_body_def.tolist()) 
with open('../FinalMatrices/R_body.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(t_body_def.tolist()) 
with open('../FinalMatrices/t_body.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(R_codo_def.tolist()) 
with open('../FinalMatrices/R_codo.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(t_codo_def.tolist()) 
with open('../FinalMatrices/t_codo.json', "w") as outfile:
    outfile.write(json_object)




    